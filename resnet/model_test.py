import os
import time
import matplotlib
matplotlib.use("Agg")  # 无界面环境也能保存图
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

from read_data_test import test_loader
from utils import compute_mae, compute_mape, compute_kge, compute_nse, compute_r2
import config
import Net

# ====== 基本配置======
DEVICE = config.DEVICE
BATCH_SIZE = config.BATCH_SIZE

Model_path = config.model_path
Excel_path = config.Excel_path

SAVE_DIR = os.path.dirname(Model_path)
os.makedirs(SAVE_DIR, exist_ok=True)

# ====== 构建并加载模型 ======
model = Net.RainNet(in_channels=config.INPUT_CHANNELS)
model.load_state_dict(torch.load(Model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()


def evaluate_model(model, test_loader, device):
    """
    在测试集上评估：
      - 计算 MAE / MAPE / r² / NSE / KGE
      - 导出 Excel（按序号排序）
      - 画预测 vs 观测散点图（由 config 控制是否绘制以及命名）
    """
    predictions_list = []
    targets_list = []
    sequence_numbers = []

    n_images = 0
    t0 = time.time()

    with torch.no_grad():
        for features, seq_nums, targets in test_loader:
            features = features.to(device)
            targets_tensor = targets.to(device).float()  # (B,)

            preds = model(features).squeeze(1)  # (B,1) -> (B,)
            sequence_numbers.extend([int(s) for s in seq_nums])
            predictions_list.extend(preds.cpu().numpy().tolist())
            targets_list.extend(targets_tensor.cpu().numpy().tolist())

            n_images += features.size(0)

    t1 = time.time()

    # ====== 转成 torch 张量,计算指标 ======
    predictions_tensor = torch.tensor(predictions_list, device=device)
    targets_tensor = torch.tensor(targets_list, device=device)

    test_mae  = compute_mae(predictions_tensor, targets_tensor).item()
    test_mape = compute_mape(predictions_tensor, targets_tensor).item()
    test_r2   = compute_r2(predictions_tensor, targets_tensor).item()
    test_nse  = compute_nse(predictions_tensor, targets_tensor).item()
    test_kge  = compute_kge(predictions_tensor, targets_tensor).item()

    # ====== 排序 + 导出 Excel ======
    sorted_data = sorted(zip(sequence_numbers, targets_list, predictions_list))
    sorted_sequence_numbers, sorted_targets_list, sorted_predictions_list = zip(*sorted_data)

    results_df = pd.DataFrame({
        '序号': sorted_sequence_numbers,
        '观测值(mm/h)': sorted_targets_list,
        '预测值(mm/h)': sorted_predictions_list
    })
    results_df.to_excel(Excel_path, index=False)

    # ====== 画散点图：预测 vs 观测 ======
    scatter_path = None
    if getattr(config, "PLOT_TEST_SCATTER", True):
        figsize = getattr(config, "SCATTER_FIG_FIGSIZE", (6, 6))
        dpi = getattr(config, "SCATTER_FIG_DPI", 150)
        basename = getattr(config, "SCATTER_FIG_BASENAME", "scatter_pred_obs")
        text_pos = getattr(config, "SCATTER_TEXT_POS", (0.68, 0.20))
        text_fontsize = getattr(config, "SCATTER_TEXT_FONTSIZE", 10)

        plt.figure(figsize=figsize)

        # 散点：观测在 x 轴，预测在 y 轴
        plt.scatter(sorted_targets_list, sorted_predictions_list,
                    label='Prediction vs Observation')

        # 1:1 理想直线
        min_v = min(min(sorted_targets_list), min(sorted_predictions_list))
        max_v = max(max(sorted_targets_list), max(sorted_predictions_list))
        plt.plot([min_v, max_v], [min_v, max_v],
                 color='red', linestyle='-', label='Perfect Line')

        plt.title('Prediction vs Observation (mm/h)')
        plt.xlabel('Observation (mm/h)')
        plt.ylabel('Prediction (mm/h)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 指标文本框
        text_str = (f"MAE={test_mae:.2f} mm/h\n"
                    f"MAPE={test_mape:.2f}%\n"
                    f"r²={test_r2:.3f}\n"
                    f"NSE={test_nse:.3f}\n"
                    f"KGE={test_kge:.3f}")

        plt.gcf().text(
            text_pos[0],
            text_pos[1],
            text_str,
            fontsize=text_fontsize,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )

        plt.tight_layout()

        scatter_path = os.path.join(SAVE_DIR, basename + ".png")
        plt.savefig(scatter_path, dpi=dpi)
        plt.close()

    # ====== 打印整体信息 ======
    total_sec = t1 - t0
    per100 = (total_sec / max(n_images, 1)) * 100.0

    print(f"评估完成：共 {n_images} 张，耗时 {total_sec:.2f} s（约 {per100:.2f} s / 100 张）")
    print(f"Excel 已保存到：{Excel_path}")
    if scatter_path is not None:
        print(f"散点图已保存到：{scatter_path}")
    else:
        print("未生成散点图（PLOT_TEST_SCATTER=False）。")

    return test_mae, test_mape, test_r2, test_nse, test_kge


if __name__ == "__main__":
    test_mae, test_mape, test_r2, test_nse, test_kge = evaluate_model(model, test_loader, DEVICE)
    print(f'测试 MAE: {test_mae:.2f} mm/h')
    print(f'测试 MAPE: {test_mape:.2f} %')
    print(f'测试 r^2: {test_r2:.3f}')
    print(f'测试 NSE: {test_nse:.3f}')
    print(f'测试 KGE: {test_kge:.3f}')
