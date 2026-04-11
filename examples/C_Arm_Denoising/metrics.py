import torch.nn.functional as F
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val==-1:
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

@torch.no_grad()
def psnr(x_denoised: torch.Tensor, target: torch.Tensor, max_val=1.0):
    mse = torch.mean((x_denoised - target) ** 2)
    return 10 * torch.log10(max_val**2 / mse)


@torch.no_grad()
def ssim(x: torch.Tensor, y: torch.Tensor, C1=0.01**2, C2=0.03**2):
    """
    x, y: [B, 1, H, W] 归一化到 [0,1] 的图像
    """
    mu_x = F.avg_pool2d(x, 3, 1, 0)
    mu_y = F.avg_pool2d(y, 3, 1, 0)

    sigma_x  = F.avg_pool2d(x * x, 3, 1, 0) - mu_x ** 2
    sigma_y  = F.avg_pool2d(y * y, 3, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 0) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return ssim_map.mean()


def nmse( pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    计算 Normalized Mean Squared Error (NMSE)
    gt: ground truth 图像 (tensor)
    pred: 预测图像 (tensor)
    返回: 单个标量 tensor
    """
    gt = gt.float()
    pred = pred.float()
    numerator = torch.sum((gt - pred) ** 2)
    denominator = torch.sum(gt ** 2)
    return numerator / denominator

def minmax_normalize(x):
    # 按 batch 中每张图像单独归一化到 [0,1]
    B = x.shape[0]
    x_norm = []
    for i in range(B):
        xi = x[i]
        min_val = xi.min()
        max_val = xi.max()
        xi = (xi - min_val) / (max_val - min_val + 1e-8)
        x_norm.append(xi)
    return torch.stack(x_norm, dim=0)
