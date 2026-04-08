# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os

import torch
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
# from utils_cifar import ema, generate_samples, infiniteloop

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper
import argparse
import pydicom
import numpy as np

from datasets.dataset import CFM_train_dicom, CFM_validation_dicom
from examples.C_Arm_Denoising.utils_carm import validate_carm
from examples.C_Arm_Denoising.metrics import AverageMeter

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def warmup_lr(step):
    return min(step, WARMUP) / WARMUP

def dose_to_cond(
    dose_value,
    unit="mAs",
    clip_min=1.0,
    clip_max=30.0,
    use_log=False
):
    """
    将剂量值转换为条件输入：
    1. 统一单位到 mAs
    2. truncate 到 [clip_min, clip_max]
    3. 可选 log 压缩
    4. 固定 Min-Max 归一化到 [0,1]

    返回:
        cond: float in [0,1]
    """

    # 1. 转换到 mAs
    if unit == "uAs":
        dose_mAs = float(dose_value) / 1000.0
    else:
        dose_mAs = float(dose_value)

    # 2. truncate
    dose_mAs = np.clip(dose_mAs, clip_min, clip_max)

    # 3. 可选 log
    if use_log:
        dose_proc = np.log1p(dose_mAs)
        clip_min_proc = np.log1p(clip_min)
        clip_max_proc = np.log1p(clip_max)
    else:
        dose_proc = dose_mAs
        clip_min_proc = clip_min
        clip_max_proc = clip_max

    # 4. 固定归一化
    cond = (dose_proc - clip_min_proc) / (clip_max_proc - clip_min_proc + 1e-12)

    return float(cond)

def collate_fn_dicom(batch):
    src_img = torch.cat([item[0] for item in batch],axis=0)
    dst_img = torch.cat([item[1] for item in batch],axis=0)
    src_dose = torch.cat([item[2] for item in batch],axis=0)
    dst_dose = torch.cat([item[3] for item in batch],axis=0)
    return src_img, dst_img, src_dose, dst_dose

def train(args):
    print(
        "lr, total_steps, ema decay, save_step:",
        args.lr,
        args.total_steps,
        args.ema_decay,
        args.save_step,
    )

    os.makedirs(os.path.join(args.log_dir, args.model), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.model))

    train_patch_path = {'low_dose':{}, 'mid_dose':{}, 'high_dose':{}}
    valid_patch_path = {'low_dose':{}, 'mid_dose':{}, 'high_dose':{}}
    full_paths = []
    for root, dirs, files in os.walk(os.path.join(args.train_root_path, args.dataset_name)):
        for name in files:
            if name=='readme.txt':
                continue
            img_path = os.path.join(root, name)
            full_paths.append(img_path)
        full_paths = sorted(full_paths)

    all, low, mid, high = {}, {}, {}, {}
    for i in range(0, len(full_paths), 3):
        group = full_paths[i:i + 3]
        dose_list = []
        for path in group:
            ds = pydicom.dcmread(path)
            try:
                dose = ds.get((0x0018, 0x1153), None)  # Exposure in uAs
                dose = dose.value / 1000
                dose = dose_to_cond(dose)
            except:
                exposure_time = ds.get((0x0018, 0x1150), None)  # Exposure Time (ms)
                time_s = exposure_time.value / 1000
                current = ds.get((0x0018, 0x8151), None)  # X-Ray Tube Current in uA
                current_mA = current.value / 1000
                dose = current_mA * time_s
                dose = dose_to_cond(dose)

            dose_list.append((dose, path))
            all.update({path:dose})
        # 按剂量排序：低 → 中 → 高
        dose_list.sort(key=lambda x: x[0])
        (dose_low, path_low), (dose_mid, path_mid), (dose_high, path_high) = dose_list
        low.update({path_low:dose_low})
        mid.update({path_mid:dose_mid})
        high.update({path_high:dose_high})

    keys = sorted(full_paths)
    train_keys = keys[0:3] + keys[6:-3]
    val_keys = [k for k in keys if k not in train_keys]


    train_patch_path['low_dose'] = {k: all[k] for k in train_keys if k in low.keys()}
    valid_patch_path['low_dose'] = {k: all[k] for k in val_keys if k in low.keys()}
    train_patch_path['mid_dose'] = {k: all[k] for k in train_keys if k in mid.keys()}
    valid_patch_path['mid_dose'] = {k: all[k] for k in val_keys if k in mid.keys()}
    train_patch_path['high_dose'] = {k: all[k] for k in train_keys if k in high.keys()}
    valid_patch_path['high_dose'] = {k: all[k] for k in val_keys if k in high.keys()}

    trainset = CFM_train_dicom(train_patch_path, name=args.dataset_name)
    validset = CFM_validation_dicom(valid_patch_path, name=args.dataset_name)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        collate_fn=collate_fn_dicom,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=args.batch_size,
        collate_fn=collate_fn_dicom,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )


    # MODELS
    net_model = UNetModelWrapper(
        dim=(1, 512, 512),
        num_res_blocks=2,
        num_channels=args.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)  # new dropout + bs of 128

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if args.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    #################################
    #            OT-CFM
    #################################

    sigma = 0.0
    if args.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif args.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif args.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif args.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {args.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
        )

    savedir = args.output_dir + args.model + "/"
    os.makedirs(savedir, exist_ok=True)

    lossesMeter = AverageMeter(name='TrainMeter total loss ')
    psnrMeter = AverageMeter(name='ValMeter PSNR')
    ssimMeter = AverageMeter(name='ValMeter SSIM')

    train_iter = iter(trainloader)
    # while (step < args.total_steps):
    with trange(args.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            try:
                batch = next(train_iter)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                train_iter = iter(trainloader)
                batch = next(train_iter)

            x1 = batch[1].to(device)
            x0 = batch[0].to(device) # x0 = torch.randn_like(x1)
            cond = batch[2].to(device)
            if cond.dim() == 1:
                cond = cond.unsqueeze(1)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

            vt = net_model(t, xt, cond=cond)

            loss = torch.mean((vt - ut) ** 2)

            loss.backward()

            lossesMeter.update(loss.item())
            if step % 200 == 0:
                writer.add_scalar("train/loss", scalar_value=lossesMeter.avg, global_step=step+1)

            torch.nn.utils.clip_grad_norm_(net_model.parameters(), args.grad_clip)  # new

            optim.step()
            sched.step()

            # ema(net_model, ema_model, args.ema_decay)  # new

            # sample and Saving the weights
            if args.save_step > 0 and step % args.save_step == 0:
                # generate_samples(net_model, args.parallel, savedir, step, net_="normal")
                # generate_samples(ema_model, args.parallel, savedir, step, net_="ema")
                val_length = len(validset)

                validate_carm(net_model, validloader, savedir, step, val_length, device, writer, psnrMeter, ssimMeter, net_="normal")
                validate_carm(ema_model, validloader, savedir, step, val_length, device, writer, psnrMeter, ssimMeter, net_="ema")

                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    savedir + f"{args.model}_carm_weights_step_{step}.pt",
                )

    writer.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_root_path",
        type=str,
        default="/data"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="dicom_for_comparing"
    )

    # Model
    parser.add_argument("--model", type=str, default="otcfm",
                        help="flow matching model type")
    parser.add_argument("--output_dir", type=str, default="./results/",
                        help="output_directory")
    parser.add_argument("--log_dir", type=str, default="./logs/",
                        help="logs_directory")

    # UNet
    parser.add_argument("--num_channel", type=int, default=128,
                        help="base channel of UNet")

    # Training
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="target learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="gradient norm clipping")
    parser.add_argument("--total_steps", type=int, default=400001,
                        help="total training steps")
    parser.add_argument("--warmup", type=int, default=5000,
                        help="learning rate warmup")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="workers of Dataloader")
    parser.add_argument("--ema_decay", type=float, default=0.9999,
                        help="ema decay rate")
    parser.add_argument("--parallel", action="store_true",
                        help="multi gpu training")

    # Evaluation
    parser.add_argument("--save_step", type=int, default=20000,
                        help="frequency of saving checkpoints, 0 to disable during training")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # app.run(train)
    args = get_args()
    global WARMUP
    WARMUP = args.warmup

    # args.dataset_name = 'dicom_for_comparing'
    # args.batch_size = 8
    # args.num_workers = 16
    # args.model = "otcfm" # icfm, fm, si
    # args.save_step = 10000 # 10k

    train(args)