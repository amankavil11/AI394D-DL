import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import DetectionMetric

def train(exp_dir: str = "logs",
          model_name: str = "detector",
          dataset_path: str = "drive_data",
          num_epoch: int = 60,
          lr: float = 1e-3,
          batch_size: int = 64,
          weight_decay: float = 1e-4,
          seed: int = 2024):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name).to(device)
    model.train()

    train_data = load_data(f"{dataset_path}/train", batch_size=batch_size, shuffle=True, pin_memory=True)
    val_data = load_data(f"{dataset_path}/val", batch_size=batch_size, pin_memory=True)

    # create loss function and optimizer
    segment_loss_func = torch.nn.CrossEntropyLoss()
    regressor_loss_func = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    global_step = 0
    train_metrics = DetectionMetric()
    val_metrics = DetectionMetric()

    for epoch in range(num_epoch):

        model.train()

        for i, map in enumerate(train_data):
            img, true_depth, track = map['image'].to(device), map['depth'].to(device), map['track'].to(device) 

            optimizer.zero_grad()

            logits, depth = model(img)
            preds = torch.argmax(logits, dim=1)
            train_metrics.add(preds, track, depth, true_depth)

            
            loss = segment_loss_func(logits, track) + regressor_loss_func(depth, true_depth)
            loss.backward()

            optimizer.step()

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for map in val_data:
                img, true_depth, track = map['image'].to(device), map['depth'].to(device), map['track'].to(device) 

                prediction, depth = model.predict(img)
                val_metrics.add(prediction, track, depth, true_depth)

                # metrics['val_acc'].append((prediction == label).float().mean().item())
    

        # log average train and val accuracy to tensorboard
        # epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        # epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        
        # logger.add_scalar("train_accuracy", epoch_train_acc, global_step)
        # logger.add_scalar("val_accuracy", epoch_val_acc, global_step)
        

         # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_metrics={train_metrics.compute()} "
                 f"val_metrics={val_metrics.compute()}"
            )
        train_metrics.reset()
        val_metrics.reset()
    

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    train()