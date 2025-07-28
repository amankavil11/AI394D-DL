from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

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
        self.first_conv_layer = torch.nn.Conv2d(in_channels, 96, kernel_size=9, stride=2, padding=4)

        self.classifier = nn.Sequential(
            nn.Conv2d(384, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        layers.append(self.first_conv_layer)
        layers.append(BlockLayer(96, 192, residual=True))
        layers.append(BlockLayer(192, 384, residual=True))
        layers.append(self.classifier)

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """

        return self.model(x).squeeze(-1).squeeze(-1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        first_out_channels: int = 64,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        class DownSampleBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1,stride = 2):
                super().__init__()
                self.initial_convs = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                    nn.ReLU(),
                )
                self.down_sample = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, stride=stride)
                

            def forward(self, x):
                out = self.initial_convs(x)
                residual = out  # save for skip connection
                out = self.down_sample(out)
                return out, residual

        class BottleneckBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
                super().__init__()
                self.bottleneck = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                    nn.ReLU(),
                )

            def forward(self, x):
                return self.bottleneck(x)


        class UpSampleBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1, stride = 2):
                super().__init__()
                self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=padding)
                
                self.concluding_convs = nn.Sequential(
                    nn.Conv2d(out_channels * 2, out_channels, kernel_size, padding=padding),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                    nn.ReLU(),
                )

            def forward(self, x, residual = None):
                out = self.up_sample(x)
                if residual is not None:
                    out = nn.functional.interpolate(out, size=residual.shape[2:], mode='bilinear', align_corners=False)
                    out = torch.cat([residual, out], dim=1)
                out = self.concluding_convs(out)
                return out

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        num_down_layers = num_up_layers = 3

        for i in range(num_down_layers):
            if i == 0:
                in_ch = in_channels
                out_ch = first_out_channels
            else:
                in_ch = first_out_channels * (2 ** (i - 1))
                out_ch = first_out_channels * (2 ** i)
            
            layer = DownSampleBlock(in_ch, out_channels=out_ch)
            self.down_layers.append(layer)

        self.bottleneck = BottleneckBlock(first_out_channels * (2 ** (num_down_layers - 1)), first_out_channels * (2 ** num_down_layers))

        for i in range(num_up_layers):
            in_ch = first_out_channels * (2 ** (num_up_layers - i ))
            out_ch = int(first_out_channels * (2 ** (num_up_layers - (i + 1))))
            self.up_layers.append(UpSampleBlock(in_ch, out_channels=out_ch))

        self.classifier = nn.Conv2d(first_out_channels, num_classes, kernel_size=1)
        self.depth_regressor = nn.Conv2d(first_out_channels, 1, kernel_size=1)
        

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        h_in, w_in = x.shape[2:] 

        residuals = []
        for layer in self.down_layers:
            x, residual = layer(x)
            residuals.append(residual)

        x = self.bottleneck(x)

        for layer in self.up_layers:
            x = layer(x, residual=residuals.pop() if residuals else None)

        logits = self.classifier(x)
        logits = logits[:, :, :h_in, :w_in]

        depth = self.depth_regressor(x).squeeze(1)
        depth = torch.sigmoid(depth[:, :h_in, :w_in])

        return logits, depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
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
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
