import os
import re
import json
from typing import List
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config

# ============ 基本配置 ============
root_dir = config.root_dir
BATCH_SIZE = config.BATCH_SIZE
INPUT_CHANNELS = config.INPUT_CHANNELS
NORM_STATS_FILE = getattr(
    config,
    "NORM_STATS_FILE",
    os.path.join(os.path.dirname(__file__), "norm_stats.json"),
)

BINARY_NORM_POLICY = getattr(config, "BINARY_NORM_POLICY", "compute")
BINARY_FIXED_MEAN_STD = getattr(config, "BINARY_FIXED_MEAN_STD", ([0.5], [0.5]))
INPUT_SIZE = getattr(config, "INPUT_SIZE", (224, 224))

# 文件名解析（与 read_data.py 保持一致）
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


def _load_norm_stats():
    if not os.path.isfile(NORM_STATS_FILE):
        raise FileNotFoundError(
            f"[Norm] 测试需要的归一化文件不存在：{NORM_STATS_FILE}，请先在训练阶段完成统计。"
        )
    with open(NORM_STATS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    mean = data.get("mean", None)
    std = data.get("std", None)
    if mean is None or std is None:
        raise ValueError(f"[Norm] 归一化文件格式不正确：{NORM_STATS_FILE}")
    print(f"[Norm] 测试阶段使用训练阶段的 mean/std: mean={mean}, std={std}")
    return mean, std


def build_transform(mean, std, in_channels: int):
    t_list = []
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


class TestDataset(Dataset):
    def __init__(self, root_dir: str, in_channels: int, transform=None):
        self.root_dir = root_dir
        self.in_channels = in_channels
        self.transform = transform

        files: List[str] = []
        for fname in os.listdir(root_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")) and parse_filename(fname) is not None:
                files.append(os.path.join(root_dir, fname))
        self.file_paths = sorted(files)

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

        # 测试阶段返回 (图像, 序号, 强度)
        return img, int(seq), float(intensity)


_mean, _std = _load_norm_stats()
_transform = build_transform(_mean, _std, INPUT_CHANNELS)
dataset = TestDataset(root_dir, in_channels=INPUT_CHANNELS, transform=_transform)
test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
