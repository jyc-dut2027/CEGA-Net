import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from read_data import train_loader, val_loader
from utils import compute_mae, compute_mape, compute_kge, compute_nse, compute_r2
import Net
import config

# =============== 基本配置 ===============
DEVICE = config.DEVICE
BATCH_SIZE = config.BATCH_SIZE
NUM_EPOCHS = config.NUM_EPOCHS
model_path = config.model_path

USE_VALIDATION = getattr(config, "USE_VALIDATION", True)

# =============== 构建模型与优化器 ===============
model = Net.RainNet(in_channels=config.INPUT_CHANNELS).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-3
)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
criterion = torch.nn.HuberLoss(delta=getattr(config, "HUBER_BETA", 1.0))

def _batch_losses(predictions, targets):

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    valid = ~torch.isnan(predictions) & ~torch.isnan(targets)
    predictions = predictions[valid]
    targets = targets[valid]


    mae = compute_mae(predictions, targets)
    mape = compute_mape(predictions, targets)
    r2 = compute_r2(predictions, targets)
    nse = compute_nse(predictions, targets)
    kge = compute_kge(predictions, targets)

    loss = criterion(predictions, targets)

    return loss, (mae.item(), mape.item(), r2.item(), nse.item(), kge.item())


@torch.no_grad()
def evaluate_on_loader(model, loader, device):
    """
    在给定 DataLoader 上评估平均 Loss 和各指标。
    """
    model.eval()
    total_loss_val = 0.0
    total_mae = total_mape = total_r2 = total_nse = total_kge = 0.0
    n_batches = 0

    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).view(-1, 1).float()

        preds = model(features).float()
        loss, (mae, mape, r2, nse, kge) = _batch_losses(preds, targets)

        total_loss_val += float(loss.item())
        total_mae += mae
        total_mape += mape
        total_r2 += r2
        total_nse += nse
        total_kge += kge
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0, 0, 0, 0, 0

    return (
        total_loss_val / n_batches,
        total_mae / n_batches,
        total_mape / n_batches,
        total_r2 / n_batches,
        total_nse / n_batches,
        total_kge / n_batches,
    )


def save_loss_curves(train_losses, val_losses, epochs, save_dir):

    if not getattr(config, "PLOT_LOSS_CURVE", True):
        return

    os.makedirs(save_dir, exist_ok=True)

    figsize = getattr(config, "LOSS_CURVE_FIGSIZE", (6, 4))
    dpi = getattr(config, "LOSS_CURVE_DPI", 150)
    basename = getattr(config, "LOSS_CURVE_BASENAME", "loss_curve")

    plt.figure(figsize=figsize)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    if val_losses:
        plt.plot(epochs, val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(save_dir, basename + ".png")
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"[Plot] 损失曲线已保存到: {out_path}")


def train_model(model, num_epochs, train_loader, val_loader, optimizer, device, scheduler=None):

    epoch_list, train_loss_hist, val_loss_hist = [], [], []
    best_metric = float('inf')
    best_model_path = model_path

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        try:
            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                colour=getattr(config, "PROGRESS_COLOR", "red"),
                desc=f"Epoch {epoch + 1}/{num_epochs}"
            )
        except TypeError:
            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{num_epochs}"
            )

        for batch_idx, (features, targets) in pbar:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).view(-1, 1).float()

            optimizer.zero_grad()
            preds = model(features).float()
            loss, _ = _batch_losses(preds, targets)

            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()

            epoch_losses.append(loss.item())

        # —— 训练平均
        avg_tr_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        train_loss_hist.append(avg_tr_loss)
        epoch_list.append(epoch + 1)

        if val_loader is not None:
            avg_val_loss, v_mae, v_mape, v_r2, v_nse, v_kge = evaluate_on_loader(model, val_loader, device)
            val_loss_hist.append(avg_val_loss)

            print(
                f"[Epoch {epoch+1}/{num_epochs}] "
                f"TrainLoss={avg_tr_loss:.4f} | ValLoss={avg_val_loss:.4f} | "
                f"Val(MAE/MAPE/r2/NSE/KGE)=({v_mae:.2f}/{v_mape:.2f}/{v_r2:.3f}/{v_nse:.3f}/{v_kge:.3f})"
            )

            # 按验证集 loss 选最优
            if avg_val_loss < best_metric:
                best_metric = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> 保存当前最优模型 @ epoch {epoch+1}")

            if scheduler is not None:
                scheduler.step(avg_val_loss)
        else:
            print(f"[Epoch {epoch+1}/{num_epochs}] TrainLoss={avg_tr_loss:.4f} | (No Val)")

            if avg_tr_loss < best_metric:
                best_metric = avg_tr_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> 保存当前最优模型 @ epoch {epoch+1}")

            if scheduler is not None:
                scheduler.step(avg_tr_loss)

    save_loss_curves(train_loss_hist, val_loss_hist, epoch_list, os.path.dirname(model_path))
    return best_metric


if __name__ == "__main__":
    t0 = time.time()

    best_metric = train_model(
        model,
        NUM_EPOCHS,
        train_loader,
        val_loader,
        optimizer,
        DEVICE,
        scheduler=scheduler
    )

    print(
        f"\n[Done] best_metric={best_metric:.4f} | "
        f"用 {'ValLoss' if USE_VALIDATION and val_loader is not None else 'TrainLoss'} "
        f"选择并保存到了: {model_path}"
    )

    elapsed = (time.time() - t0) / 60.0
    print(f"\n[All Done] total time = {elapsed:.2f} min | 最终模型权重: {model_path}")
