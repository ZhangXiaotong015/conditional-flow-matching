import os
import copy
import torch
from torch import nn
from torchdyn.core import NeuralODE
from torchdyn.numerics.odeint import odeint
from torchvision.utils import save_image
from examples.C_Arm_Denoising.metrics import AverageMeter, psnr, ssim, nmse

@torch.no_grad()
def validate_carm(model, validloader, savedir, step, val_length, device, writer, logging, psnrMeter, ssimMeter, nmseMeter, net_="normal", condition=None):
    model.eval()

    # for batch in val_loader:
    step_val = 0
    val_iter = iter(validloader)
    while (step_val < val_length):
        try:
            batch = next(val_iter)
        except StopIteration:
            val_iter = iter(validloader)
            batch = next(val_iter)

        x_noisy = batch[0].to(device)      # [B, 1, H, W]
        if condition is True:
            cond    = batch[2].to(device)    # [B, cond_dim]
        else:
            cond = None
        target = batch[1].to(device)

        # 为当前 batch 创建一个 ODE 函数（cond 是动态的）
        class WrappedODEFunc(torch.nn.Module):
            def __init__(self, net, cond):
                super().__init__()
                self.net = net
                self.cond = cond

            def forward(self, t, x, args=None):
                return self.net(t, x, self.cond)

        ode_func = WrappedODEFunc(model, cond)

        # node = NeuralODE(ode_func, solver="euler", sensitivity="adjoint")
        node = NeuralODE(ode_func, solver="rk4", sensitivity="adjoint")

        # t_span = torch.linspace(1, 0, 100, device=device)
        t_span = torch.linspace(1, 0, 30, device=device)

        traj = node.trajectory(x_noisy, t_span)
        # traj = odeint(ode_func, x_noisy, t_span, solver="rk4")0

        x_denoised = traj[-1]

        psnr_val = psnr(x_denoised, target, max_val=1.0)
        ssim_val = ssim(x_denoised, target)
        nmse_val = nmse(x_denoised, target)

        psnrMeter.update(psnr_val.item())
        ssimMeter.update(ssim_val.item())
        nmseMeter.update(nmse_val.item())
        if step % 10000 == 0 and step_val==val_length-1:
            logging.info(f"Step {step}, Validation PSNR: {psnrMeter.avg:.4f}, SSIM: {ssimMeter.avg:.4f}, NMSE: {nmseMeter.avg:.4f}")
            writer.add_scalar("validation/PSNR", scalar_value=psnrMeter.avg, global_step=step + 1)
            writer.add_scalar("validation/SSIM", scalar_value=ssimMeter.avg, global_step=step + 1)
            writer.add_scalar("validation/NMSE", scalar_value=nmseMeter.avg, global_step=step + 1)

        # 反归一化（根据你的 preprocessing）
        # x_out = torch.clamp(x_denoised, 0, 1) * 65535.0

        # 保存可视化
        B = x_denoised.shape[0]
        x_out_vis = torch.clamp(x_denoised, 0, 1)
        # target_vis = torch.clamp(target, 0, 1)
        combined = torch.cat([x_noisy, x_out_vis, target], dim=0)

        save_image(
            combined,
            os.path.join(savedir, f"{net_}_val_iter_{step}_step_{step_val}.png"),
            nrow=B,
            normalize = False
        )

        step_val += 1

    model.train()
