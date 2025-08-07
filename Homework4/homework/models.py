from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.mlp_net = nn.Sequential(
            nn.Linear(in_features=4*n_track, out_features=512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(in_features=512, out_features=512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(in_features=512, out_features=256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(in_features=256, out_features=128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(in_features=128, out_features=2*n_waypoints)
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        track = torch.cat([track_left, track_right], dim=2)
        x = track.view(track.size(0), -1)
        out = self.mlp_net(x)
        return out.view(-1, self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 128,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.num_layers = 2
        self.num_heads = int(d_model // 64)

        self.query_embed = nn.Embedding(n_waypoints, d_model)
        self.input_proj = nn.Linear(4, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=self.num_heads, 
            dim_feedforward=4 * d_model, 
            activation="gelu", 
            dropout=0.2,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        track = torch.cat([track_left, track_right], dim=2)

        memory = self.input_proj(track).to(track.device)

        queries = self.query_embed.weight.expand(memory.size(0), -1, -1).to(track.device)  

        decoded = self.decoder(tgt=queries, memory=memory)

        return self.output_proj(decoded)


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)


        class BlockLayer(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_sizes=[3, 1, 3], stride=2, residual=False):
                super().__init__()
                self.residual = residual

                self.conv_block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_sizes[0], stride, padding=kernel_sizes[0] // 2),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_sizes[1], stride=1, padding=kernel_sizes[1] // 2),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_sizes[2], stride=1, padding=kernel_sizes[2] // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
                if self.residual:
                    self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
                else:
                    self.identity = None


            def forward(self, x):
                out = self.conv_block(x)

                if self.residual:
                    return out + self.identity(x)
                else:
                    return out


        layers = []
        self.first_conv_layer = torch.nn.Conv2d(3, 96, kernel_size=9, stride=2, padding=4)

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(384, 2*self.n_waypoints, kernel_size=1),
        )

        layers.append(self.first_conv_layer)
        layers.append(BlockLayer(96, 192, residual=True))
        layers.append(BlockLayer(192, 384, residual=True))
        layers.append(self.regressor)

        self.model = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        return self.model(x).reshape(x.size(0), self.n_waypoints, 2)

        


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
