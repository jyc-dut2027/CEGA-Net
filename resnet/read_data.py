import os
import re
import json
import random
from typing import List, Tuple
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config

# ================== 基本配置 ==================
root_dir = config.root_dir
BATCH_SIZE = config.BATCH_SIZE
INPUT_CHANNELS = config.INPUT_CHANNELS

# 均值 / 方差统计与缓存
NORM_STATS_FILE = getattr(
    config,
    "NORM_STATS_FILE",
    os.path.join(os.path.dirname(__file__), "norm_stats.json"),
)
NORM_SCOPE = getattr(config, "NORM_SCOPE", "all")
NORM_USE_CACHE = getattr(config, "NORM_USE_CACHE", True)
FORCE_REBUILD_NORM = getattr(config, "FORCE_REBUILD_NORM", False)
MAX_STAT_IMAGES = getattr(config, "MAX_STAT_IMAGES", None)

# 二值 / 灰度归一化策略
BINARY_NORM_POLICY = getattr(config, "BINARY_NORM_POLICY", "compute")
BINARY_FIXED_MEAN_STD = getattr(config, "BINARY_FIXED_MEAN_STD", ([0.5], [0.5]))

# 输入尺寸 (H, W)
INPUT_SIZE = getattr(config, "INPUT_SIZE", (224, 224))

# 验证集占比
USE_VALIDATION = getattr(config, "USE_VALIDATION", True)
VAL_RATIO = getattr(config, "VAL_RATIO", 0.25)

# ================== 文件名解析 ==================
# 支持两种格式：
#   00006_1_6.5.jpg -> (seq=6, event_id=1, intensity=6.5)
#   2_6.jpg         -> (seq=2, event_id=-1, intensity=6.0)
RE_NEW = re.compile(r"^(\d+)_([0-9]+)_([0-9.]+)\.(jpg|jpeg|png)$", re.IGNORECASE)
RE_OLD = re.compile(r"^(\d+)_([0-9.]+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_filename(fname: str):
    m = RE_NEW.match(fname)
    if m:
        seq = int(m.group(1))
        event_id = int(m.group(2))
        intensity = float(m.group(3))
        return seq, event_id, intensity
    m2 = RE_OLD.match(fname)
    if m2:
        seq = int(m2.group(1))
        event_id = -1
        intensity = float(m2.group(2))
        return seq, event_id, intensity
    return None


# ================== 统计均值 / 方差 ==================
def _compute_mean_std(image_paths: List[str], in_channels: int) -> Tuple[List[float], List[float]]:
    if not image_paths:
        if in_channels == 1:
            return [0.5], [0.25]
        else:
            return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    paths = list(image_paths)
    if MAX_STAT_IMAGES is not None and len(paths) > MAX_STAT_IMAGES:
        rng = random.Random(3407)
        paths = rng.sample(paths, MAX_STAT_IMAGES)

    if in_channels == 1:
        sum_v = 0.0
        sum2_v = 0.0
        count = 0
    else:
        sum_v = np.zeros(3, dtype=np.float64)
        sum2_v = np.zeros(3, dtype=np.float64)
        count = 0

    for p in paths:
        img = Image.open(p)
        if in_channels == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        img = img.resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0

        if in_channels == 1:
            v = arr.reshape(-1)
            sum_v += v.sum()
            sum2_v += (v * v).sum()
            count += v.size
        else:
            v = arr.reshape(-1, 3)
            sum_v += v.sum(axis=0)
            sum2_v += (v * v).sum(axis=0)
            count += v.shape[0]

    if count == 0:
        if in_channels == 1:
            return [0.5], [0.25]
        else:
            return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if in_channels == 1:
        mean = sum_v / count
        var = max(sum2_v / count - mean * mean, 1e-8)
        std = float(np.sqrt(var))
        mean_list = [float(mean)]
        std_list = [std]
    else:
        mean = sum_v / count
        var = np.maximum(sum2_v / count - mean * mean, 1e-8)
        std = np.sqrt(var)
        mean_list = [float(x) for x in mean.tolist()]
        std_list = [float(x) for x in std.tolist()]

    print(f"[Norm] 重新统计完成: mean={mean_list}, std={std_list}")
    return mean_list, std_list


def _load_or_create_norm_stats(image_paths: List[str], in_channels: int):
    os.makedirs(os.path.dirname(NORM_STATS_FILE), exist_ok=True)

    if NORM_USE_CACHE and (not FORCE_REBUILD_NORM) and os.path.isfile(NORM_STATS_FILE):
        try:
            with open(NORM_STATS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            mean = data.get("mean", None)
            std = data.get("std", None)
            if mean is not None and std is not None:
                print(f"[Norm] 使用缓存文件: {NORM_STATS_FILE}")
                return mean, std
        except Exception as e:
            print(f"[Norm] 读取缓存失败，将重新统计: {e}")

    mean, std = _compute_mean_std(image_paths, in_channels)
    try:
        with open(NORM_STATS_FILE, "w", encoding="utf-8") as f:
            json.dump({"mean": mean, "std": std}, f, ensure_ascii=False, indent=2)
        print(f"[Norm] 缓存已写入: {NORM_STATS_FILE}")
    except Exception as e:
        print(f"[Norm] 写入缓存失败: {e}")
    return mean, std


# ================== Transform 构建 ==================
def build_transform(mean, std, in_channels: int):
    t_list = []
    # Dataset 用 PIL 读图，因此不需要 ToPILImage，直接 Resize
    t_list.append(transforms.Resize(INPUT_SIZE))
    t_list.append(transforms.ToTensor())

    if in_channels == 1:
        if BINARY_NORM_POLICY == "skip":
            pass
        elif BINARY_NORM_POLICY == "fixed":
            fixed_mean, fixed_std = BINARY_FIXED_MEAN_STD
            t_list.append(transforms.Normalize(mean=fixed_mean, std=fixed_std))
        else:
            t_list.append(transforms.Normalize(mean=mean, std=std))
    else:
        t_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(t_list)


# ================== Dataset ==================
class RainDataset(Dataset):
    def __init__(self, file_paths: List[str], in_channels: int, transform=None):
        self.file_paths = file_paths
        self.in_channels = in_channels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        p = self.file_paths[idx]
        fname = os.path.basename(p)
        info = parse_filename(fname)
        if info is None:
            raise ValueError(f"无法从文件名解析序号与强度: {fname}")
        seq, event_id, intensity = info

        img = Image.open(p)
        if self.in_channels == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # 训练阶段返回 (图像, 强度)
        return img, float(intensity)

# ================== 构建文件列表 ==================
all_files: List[str] = []
for fname in os.listdir(root_dir):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")) and parse_filename(fname) is not None:
        all_files.append(os.path.join(root_dir, fname))
all_files = sorted(all_files)

mean, std = _load_or_create_norm_stats(all_files, INPUT_CHANNELS)
_transform = build_transform(mean, std, INPUT_CHANNELS)

# ================== 划分 train / val ==================
if USE_VALIDATION:
    rng = random.Random(2025)
    paths = list(all_files)
    rng.shuffle(paths)
    n_total = len(paths)
    n_val = int(n_total * VAL_RATIO)
    val_files = paths[:n_val]
    train_files = paths[n_val:]
else:
    train_files = list(all_files)
    val_files = []

train_dataset = RainDataset(train_files, INPUT_CHANNELS, transform=_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

if USE_VALIDATION and val_files:
    val_dataset = RainDataset(val_files, INPUT_CHANNELS, transform=_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
else:
    val_loader = None

