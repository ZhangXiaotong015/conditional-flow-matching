import os
import copy
import torch
from torch import nn
from torchdyn.core import NeuralODE
from torchdyn.numerics.odeint import odeint
from torchvision.utils import save_image


@torch.no_grad()
def validate_carm(model, val_iter, savedir, step, val_length, device, net_="normal"):
    model.eval()

    # for batch in val_loader:
    step_val = 0
    while (step_val < val_length):
        batch = next(val_iter)
        x_noisy = batch[0].to(device)      # [B, 1, H, W]
        cond    = batch[2].to(device)    # [B, cond_dim]

        # 为当前 batch 创建一个 ODE 函数（cond 是动态的）
        class WrappedODEFunc(torch.nn.Module):
            def __init__(self, net, cond):
                super().__init__()
                self.net = net
                self.cond = cond

            def forward(self, t, x, args=None):
                return self.net(t, x, self.cond)

        ode_func = WrappedODEFunc(model, cond)

        # node = NeuralODE(ode_func, solver="euler")
        node = NeuralODE(ode_func, solver="rk4")

        # t_span = torch.linspace(1, 0, 100, device=device)
        t_span = torch.linspace(1, 0, 5, device=device)

        traj = node.trajectory(x_noisy, t_span)
        # traj = odeint(ode_func, x_noisy, t_span, solver="rk4")0

        x_denoised = traj[-1]
        raise ValueError(x_denoised.shape)
        # 反归一化（根据你的 preprocessing）
        x_out = x_denoised.clamp(-1, 1)
        x_out = (x_out + 1) / 2

        # 保存可视化
        save_image(
            x_out,
            os.path.join(savedir, f"val_step{step}_batch{i}.png"),
            nrow=B
        )

        step_val += 1

    model.train()
