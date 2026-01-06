import os
import torch

# ================== 基础开关 ==================
# True -> 评估/测试模式；False -> 训练模式
train_or_test = False  # 训练时改为 False，测试/评估时改为 True
# 统一的网络输入尺寸（H, W），用于 Resize / 归一化统计
INPUT_SIZE = (224, 224)

PROGRESS_COLOR = "red"

# ================== 通道与输入尺寸 ==================
# 输入通道数：1（灰度 / 二值） 或 3（RGB）
INPUT_CHANNELS = 3

# ================== 模型与统计文件路径 ==================

DATA_ROOT = r'model_path/zhengdata'

EXP_ID = "1"
# 归一化统计文件（按通道数区分）
CH_TAG = f"{INPUT_CHANNELS}ch"
NORM_STATS_FILE = os.path.join(DATA_ROOT, f"norm_stats_{CH_TAG}.json")

# 模型和 Excel 名称（由 EXP_ID 统一控制）
MODEL_NAME = f"best{EXP_ID}_model.pth"      # 例如 best1_model.pth
EXCEL_NAME = f"test{EXP_ID}_results.xlsx"   # 例如 test1_results.xlsx

# ================== 归一化策略（训练 & 测试共用） ==================

NORM_SCOPE = "all"

# 是否使用缓存（norm_stats 文件），若训练集文件集合变化会自动重算
NORM_USE_CACHE = True

# 强制重算一次时置 True（跑完会重新写入 json）
FORCE_REBUILD_NORM = False

# 单通道（二值/灰度）归一化策略：
#   "compute" -> 按数据统计 mean/std（默认）
#   "skip"    -> 不做 Normalize（保持 0~1）
#   "fixed"   -> 使用固定 mean/std
BINARY_NORM_POLICY = "compute"
BINARY_FIXED_MEAN_STD = ([0.5], [0.5])
BINARY_NEAREST_RESIZE = False
MAX_STAT_IMAGES = None

# ================== 设备 ==================
DEVICE = torch.device('cuda')

# ================== 可视化相关配置 ==================
# —— 训练损失曲线 —— #
PLOT_LOSS_CURVE = True                        # 是否绘制并保存训练/验证损失曲线
LOSS_CURVE_BASENAME = f"loss{EXP_ID}"         # 例如 loss1_stage1.png
LOSS_CURVE_FIGSIZE = (6, 4)                   # 图像尺寸 (width, height)
LOSS_CURVE_DPI = 150                          # 分辨率

# —— 测试散点图（预测 vs 观测） —— #
PLOT_TEST_SCATTER = True                      # 是否绘制并保存散点图
SCATTER_FIG_BASENAME = f"predicition{EXP_ID}"  # 例如 prediction1.png
SCATTER_FIG_FIGSIZE = (6, 6)                  # 图像尺寸
SCATTER_FIG_DPI = 150                         # 分辨率
SCATTER_TEXT_POS = (0.68, 0.20)               # 指标文本的相对位置（相对于整个 figure）
SCATTER_TEXT_FONTSIZE = 10                    # 文本字号

# ================== 训练 / 测试不同部分 ==================
if train_or_test:
    # ========== 测试 / 评估模式 ==========
    BATCH_SIZE = 64

    # 测试集根目录（只读）
    root_dir = r'E:\Dataset\Forcnn\zhengdata\rain\test'
    test_dir = root_dir

    # 评估时读取的模型与导出 Excel
    model_path = os.path.join(DATA_ROOT, MODEL_NAME)
    Excel_path = os.path.join(DATA_ROOT, EXCEL_NAME)

else:
    # ========== 训练模式 ==========
    BATCH_SIZE = 64
    NUM_EPOCHS = 100

    # 训练集池：已通过“划分脚本”分好 train_pool / test
    root_dir = r'E:\Dataset\Forcnn\zhengdata\rain\train_pool'

    # 训练保存的模型路径（统一叫 best{EXP_ID}_model.pth）
    model_path = os.path.join(DATA_ROOT, MODEL_NAME)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # -------- 验证集使用策略 --------
    USE_VALIDATION = True
    VAL_RATIO = 0.25

