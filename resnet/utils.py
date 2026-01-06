import torch

def compute_mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

def compute_mape(predictions, targets, eps: float = 1e-8):
    # 避免目标为 0 导致除零
    denom = torch.clamp(torch.abs(targets), min=eps)
    absolute_percentage_errors = torch.abs((targets - predictions) / denom)
    mape = torch.mean(absolute_percentage_errors) * 100.0
    return mape
def compute_nse(pred, target, eps: float = 1e-8):
    """1 - SSE / SST，SST 加 eps 防止 0。"""
    pred = pred.view(-1).float()
    target = target.view(-1).float()
    sse = torch.sum((pred - target) ** 2)
    sst = torch.sum((target - torch.mean(target)) ** 2)
    sst = torch.clamp(sst, min=eps)
    return 1.0 - sse / sst

def compute_kge(pred, target, eps: float = 1e-8):
    x = pred.view(-1).float()
    y = target.view(-1).float()
    xm, ym = torch.mean(x), torch.mean(y)
    xs, ys = torch.std(x, unbiased=False), torch.std(y, unbiased=False)

    vx = x - xm
    vy = y - ym
    # Pearson r（数值稳健）
    r_num = torch.sum(vx * vy)
    r_den = torch.sqrt(torch.sum(vx * vx) * torch.sum(vy * vy) + eps)
    r = r_num / r_den

    alpha = xs / (ys + eps)
    beta  = (xm + eps) / (ym + eps)
    kge = 1.0 - torch.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return kge
def compute_r(predictions, targets):
    """
    计算皮尔逊相关系数 r。

    参数：
        predictions (torch.Tensor): 预测值（降雨强度，mm/hr）。
        targets (torch.Tensor): 实际值（降雨强度，mm/hr）。

    返回：
        torch.Tensor: 皮尔逊相关系数 r。
    """
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    mean_pred = torch.mean(predictions)
    mean_target = torch.mean(targets)
    cov = torch.sum((predictions - mean_pred) * (targets - mean_target))
    std_pred = torch.sqrt(torch.sum((predictions - mean_pred) ** 2))
    std_target = torch.sqrt(torch.sum((targets - mean_target) ** 2))
    r = cov / (std_pred * std_target + 1e-8)
    return r

def compute_r2(predictions, targets):
    """
    计算皮尔逊相关系数的平方 r^2。

    参数：
        predictions (torch.Tensor): 预测值（降雨强度，mm/hr）。
        targets (torch.Tensor): 实际值（降雨强度，mm/hr）。

    返回：
        torch.Tensor: 皮尔逊相关系数的平方 r^2。
    """
    r = compute_r(predictions, targets)
    r2 = r ** 2
    return r2

