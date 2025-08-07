from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils import tensorboard as tb

from .models import load_model, save_model
from .metrics import PlannerMetric
from .datasets.road_dataset import load_data


def train(exp_dir: str = "logs",
        model_name: str = "cnn_planner",
        transform_pipeline: str = "state_only",
        dataset_path: str = "drive_data",
        num_epoch: int = 100,
        lr: float = 1e-4,
        batch_size: int = 128,
        weight_decay: float = 1e-4,
        seed: int = 2024
        ):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)
    scheduler = None

    model = load_model(model_name).to(device)
    model.train()

    if model_name == 'cnn_planner':
        transform_pipeline = 'default'
        train_data = load_data(f"{dataset_path}/train", batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), transform_pipeline=transform_pipeline)
        val_data = load_data(f"{dataset_path}/val", batch_size=batch_size, pin_memory=torch.cuda.is_available(), transform_pipeline=transform_pipeline)

        lr = 1e-3
        num_epoch = 60
        weight_decay = 1e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        train_data = load_data(f"{dataset_path}/train", batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), transform_pipeline=transform_pipeline)
        val_data = load_data(f"{dataset_path}/val", batch_size=batch_size, pin_memory=torch.cuda.is_available(), transform_pipeline=transform_pipeline)

    loss_func = torch.nn.L1Loss()
    if model_name == 'mlp_planner':
        lr = 0.01
        num_epoch = 20
        weight_decay = 1e-5
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif model_name == 'transformer_planner':
        lr = 5e-3
        num_epoch = 20
        weight_decay = 1e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95) #Probably don't need scheduler tbh


    global_step = 0
    train_metrics = PlannerMetric()
    val_metrics = PlannerMetric()

    for epoch in range(num_epoch):

        for batch in train_data:
            waypoints = batch['waypoints'].to(device)
            mask = batch['waypoints_mask'].to(device)

            if model_name == "cnn_planner":
                img = batch['image'].to(device)
                pred_waypoints = model(img)
            else:
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                pred_waypoints = model(track_left, track_right)
            

            optimizer.zero_grad()

            

            pred_masked = pred_waypoints[mask]
            target_waypoints = waypoints[mask]

            # Separate longitude and latitude
            pred_long = pred_masked[:, 0]
            pred_lat = pred_masked[:, 1]
            target_long = target_waypoints[:, 0]
            target_lat = target_waypoints[:, 1]

            # Compute individual losses
            long_loss = loss_func(pred_long, target_long)
            lat_loss = loss_func(pred_lat, target_lat)
            loss = 0.2*long_loss + lat_loss
            
            loss.backward()
            optimizer.step()

            train_metrics.add(pred_waypoints, waypoints, mask)
            global_step += 1
        if scheduler:
            scheduler.step()


        with torch.inference_mode():
            model.eval()
            for batch in val_data:
                waypoints = batch['waypoints'].to(device)
                mask = batch['waypoints_mask'].to(device)

                if model_name == "cnn_planner":
                    img = batch['image'].to(device)
                    pred_wp = model(img)
                else:
                    track_left = batch['track_left'].to(device)
                    track_right = batch['track_right'].to(device)
                    pred_wp = model(track_left, track_right)

                    val_metrics.add(pred_wp, waypoints, mask)

        
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            computed_train_metrics = train_metrics.compute()
            computed_val_metrics = val_metrics.compute()
            train_lat_error = computed_train_metrics['lateral_error']
            train_lon_error = computed_train_metrics['longitudinal_error']
            val_lat_error = computed_val_metrics['lateral_error']
            val_lon_error = computed_val_metrics['longitudinal_error']
            train_weighted_loss = 0.2 * train_lon_error + train_lat_error
            val_weighted_loss = 0.2 * val_lon_error + val_lat_error

            print(
                f"\nEpoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_metrics={train_lon_error, train_lat_error, train_weighted_loss} "
                 f"val_metrics={val_lon_error, val_lat_error, val_weighted_loss}"
            )
        
        train_metrics.reset()
        val_metrics.reset()
    
    save_model(model)

    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")
      

if __name__ == '__main__':
    train()
    