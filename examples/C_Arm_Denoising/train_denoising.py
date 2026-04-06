# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os

import torch
# from absl import app, flags
from torchvision import datasets, transforms
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

# FLAGS = flags.FLAGS
#
# flags.DEFINE_string("model", "otcfm", help="flow matching model type")
# flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# # UNet
# flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")
#
# # Training
# flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
# flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
# flags.DEFINE_integer(
#     "total_steps", 400001, help="total training steps"
# )  # Lipman et al uses 400k but double batch size
# flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
# flags.DEFINE_integer("batch_size", 128, help="batch size")  # Lipman et al uses 128
# flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
# flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
# flags.DEFINE_bool("parallel", False, help="multi gpu training")
#
# # Evaluation
# flags.DEFINE_integer(
#     "save_step",
#     20000,
#     help="frequency of saving checkpoints, 0 to disable during training",
# )


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def warmup_lr(step):
    return min(step, WARMUP) / WARMUP


def train(args):
    print(
        "lr, total_steps, ema decay, save_step:",
        args.lr,
        args.total_steps,
        args.ema_decay,
        args.save_step,
    )

    train_patch_path = {'full':[], 'low_dose':{}, 'mid_dose':{}, 'high_dose':{}}

    for root, dirs, files in os.walk(os.path.join(args.train_root_path, args.dataset_name)):
        for name in files:
            if name=='readme.txt':
                continue
            img_path = os.path.join(root, name)
            train_patch_path['full'].append(img_path)
        train_patch_path['full'] = sorted(train_patch_path['full'])

    full_paths = train_patch_path['full']
    low, mid, high = {}, {}, {}
    for i in range(0, len(full_paths), 3):
        group = full_paths[i:i + 3]
        dose_list = []
        for path in group:
            ds = pydicom.dcmread(path)
            try:
                dose = ds.get((0x0018, 0x1153), None)  # Exposure in uAs
                dose = dose / 1000
            except:
                exposure_time = ds.get((0x0018, 0x1150), None)  # Exposure Time (ms)
                time_s = exposure_time / 1000
                current = ds.get((0x0018, 0x8151), None)  # X-Ray Tube Current in uA
                current_mA = current / 1000
                dose = current_mA * time_s

            dose_list.append((dose, path))
        # 按剂量排序：低 → 中 → 高
        dose_list.sort(key=lambda x: x[0])
        (dose_low, path_low), (dose_mid, path_mid), (dose_high, path_high) = dose_list
        low.update({path_low:dose_low})
        mid.update({path_mid:dose_mid})
        high.update({path_high:dose_high})

    train_patch_path['low_dose'].update(low)
    train_patch_path['mid_dose'].update(mid)
    train_patch_path['high_dose'].update(high)
    raise ValueError(train_patch_path['low_dose'])
    for img_path in train_patch_path['full']:
        ds = pydicom.dcmread(img_path)
        try:
            dose = ds.get((0x0018, 0x1153), None) # Exposure in uAs
            dose_mAs = dose / 1000
        except:
            exposure_time = ds.get((0x0018, 0x1150), None) # Exposure Time (ms)
            time_s = exposure_time / 1000
            current = ds.get((0x0018, 0x8151), None) # X-Ray Tube Current in uA
            current_mA = current / 1000
            dose_mAs = current_mA * time_s

    train_patch_path['ct'] = sorted(train_patch_path['ct'])
    train_patch_path['vessel'] = sorted(train_patch_path['vessel'])
    train_patch_path['graph'] = sorted(train_patch_path['graph'])

    train_folder = sorted(train_folder)
    logger.log("cases in the training set: "+str(train_folder))

    trainset = Dataset3D_diffusionGAT_train_LiVS(train_patch_path, name='train')


    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)

    # MODELS
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)  # new dropout + bs of 128

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
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
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
        )

    savedir = FLAGS.output_dir + FLAGS.model + "/"
    os.makedirs(savedir, exist_ok=True)

    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).to(device)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)  # new

            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                generate_samples(net_model, FLAGS.parallel, savedir, step, net_="normal")
                generate_samples(ema_model, FLAGS.parallel, savedir, step, net_="ema")
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    savedir + f"{FLAGS.model}_cifar10_weights_step_{step}.pt",
                )


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

    train(args)