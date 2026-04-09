import torch
import numpy as np
import pydicom
import random
from PIL import Image
import SimpleITK as sitk
import torchvision.transforms as T
import torchvision.transforms.functional as F

def save_png(tensor, path):
    # tensor: [1, H, W] or [H, W]
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)

    # normalize to 0–255
    img = tensor.clone()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).byte()

    # convert to PIL and save
    pil_img = F.to_pil_image(img)
    pil_img.save(path)

def load_dicom(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)[0]  # [1,H,W] → [H,W]
    return arr

def dicom_to_8bit(img, window_min=None, window_max=None):
    """
    将16-bit DICOM图像转换为8-bit (0-255)
    可以指定窗口范围，否则自动用全局最小最大值
    """
    if window_min is None:
        window_min = np.min(img)
    if window_max is None:
        window_max = np.max(img)

    # 截断到窗口范围
    img = np.clip(img, window_min, window_max)

    # 归一化到0-255
    img_8bit = ((img - window_min) / (window_max - window_min) * 255).astype(np.uint8)
    return img_8bit

class CFM_train_dicom(torch.utils.data.Dataset):
    def __init__(self, list_IDs, name=None, pre_processing=True):
        self.list_IDs = list_IDs
        self.name = name
        self.pre_processing = pre_processing
        self.crop = T.Compose([
            T.RandomCrop(512),
        ])
        # self.resize = T.Compose([
        #     T.Resize((512, 512)),
        #     # T.ToTensor(),
        # ])
        #
        # self.crop_then_resize = T.Compose([
        #     T.RandomCrop(1024),
        #     T.Resize((512, 512)),
        #     # T.ToTensor(),
        # ])

    def __len__(self):
        return len(self.list_IDs['high_dose']) * 8

    def __getitem__(self, index):
        key = random.choice(['low_dose', 'mid_dose'])
        abs_idx = index % len(self.list_IDs['high_dose'])

        src_items = list(self.list_IDs[key].items())
        dst_items = list(self.list_IDs['high_dose'].items())

        src_path, src_dose = src_items[abs_idx]
        dst_path, dst_dose = dst_items[abs_idx]

        src_dose = np.array(round(src_dose, 2))
        dst_dose = np.array(round(dst_dose, 2))

        # src_img = pydicom.dcmread(src_path).pixel_array
        # dst_img = pydicom.dcmread(dst_path).pixel_array
        src_img = load_dicom(src_path)
        dst_img = load_dicom(dst_path)

        src_dose = torch.from_numpy(src_dose).to(torch.float32).unsqueeze(0)
        dst_dose = torch.from_numpy(dst_dose).to(torch.float32).unsqueeze(0)
        src_img = torch.from_numpy(src_img).to(torch.float32).unsqueeze(0)
        dst_img = torch.from_numpy(dst_img).to(torch.float32).unsqueeze(0)

        # src_img = torch.clamp(src_img, 0, 65535) / 65535.0
        # dst_img = torch.clamp(dst_img, 0, 65535) / 65535.0
        src_img = (src_img - src_img.mean()) / src_img.std()
        dst_img = (dst_img - dst_img.mean()) / dst_img.std()

        if self.pre_processing:
            src_dst_img = self.crop(torch.cat((src_img, dst_img), dim=0))
            src_img = src_dst_img[0].unsqueeze(0).unsqueeze(0)
            dst_img = src_dst_img[1].unsqueeze(0).unsqueeze(0)
            # _, H, W = src_img.shape
            # if H <= 1024 and W <= 1024:
            #     src_dst_img = self.resize(torch.cat((src_img, dst_img),dim=0))
            #     src_img = src_dst_img[0].unsqueeze(0).unsqueeze(0)
            #     dst_img = src_dst_img[1].unsqueeze(0).unsqueeze(0)
            # else:
            #     src_dst_img = self.crop_then_resize(torch.cat((src_img, dst_img),dim=0))
            #     src_img = src_dst_img[0].unsqueeze(0).unsqueeze(0)
            #     dst_img = src_dst_img[1].unsqueeze(0).unsqueeze(0)

        # save_png(src_img, "/scratch/conditional-flow-matching/src.png")
        # save_png(dst_img, "/scratch/conditional-flow-matching/dst.png")

        return tuple((src_img, dst_img, src_dose, dst_dose))

class CFM_validation_dicom(torch.utils.data.Dataset):
    def __init__(self, list_IDs, name=None, pre_processing=True):
        self.list_IDs = list_IDs
        self.name = name
        self.pre_processing = pre_processing

        self.crop = T.Compose([
            T.RandomCrop(512),
        ])

    def __len__(self):
        return len(self.list_IDs['high_dose']) * 4

    def __getitem__(self, index):
        key = random.choice(['low_dose', 'mid_dose'])
        abs_idx = index % len(self.list_IDs['high_dose'])

        src_items = list(self.list_IDs[key].items())
        dst_items = list(self.list_IDs['high_dose'].items())

        src_path, src_dose = src_items[abs_idx]
        dst_path, dst_dose = dst_items[abs_idx]

        src_dose = np.array(round(src_dose, 2))
        dst_dose = np.array(round(dst_dose, 2))

        # src_img = pydicom.dcmread(src_path).pixel_array
        # dst_img = pydicom.dcmread(dst_path).pixel_array
        src_img = load_dicom(src_path)
        dst_img = load_dicom(dst_path)

        src_dose = torch.from_numpy(src_dose).to(torch.float32).unsqueeze(0).unsqueeze(0)
        dst_dose = torch.from_numpy(dst_dose).to(torch.float32).unsqueeze(0).unsqueeze(0)
        src_img = torch.from_numpy(src_img).to(torch.float32).unsqueeze(0)
        dst_img = torch.from_numpy(dst_img).to(torch.float32).unsqueeze(0)

        # src_img = torch.clamp(src_img, 0, 65535) / 65535.0
        # dst_img = torch.clamp(dst_img, 0, 65535) / 65535.0
        src_img = (src_img - src_img.mean()) / src_img.std()
        dst_img = (dst_img - dst_img.mean()) / dst_img.std()

        if self.pre_processing:
            # _, H, W = src_img.shape
            src_dst_img = self.crop(torch.cat((src_img, dst_img),dim=0))
            src_img = src_dst_img[0].unsqueeze(0).unsqueeze(0)
            dst_img = src_dst_img[1].unsqueeze(0).unsqueeze(0)

        # save_png(src_img, "/scratch/conditional-flow-matching/src.png")
        # save_png(dst_img, "/scratch/conditional-flow-matching/dst.png")

        return tuple((src_img, dst_img, src_dose, dst_dose))


class CFM_train_jpeg(torch.utils.data.Dataset):
    def __init__(self, list_IDs, name=None, pre_processing=None):
        self.list_IDs = list_IDs
        self.name = name
        self.pre_processing = pre_processing
        self.crop = T.Compose([
            T.RandomCrop(256),
        ])

    def __len__(self):
        return len(self.list_IDs['high_dose'])

    def __getitem__(self, index):

        src_path = self.list_IDs['low_dose'][index]
        dst_path = self.list_IDs['high_dose'][index]

        src_img = np.array(Image.open(src_path))  # shape: (256,256) value range: [0,255]
        dst_img = np.array(Image.open(dst_path))

        src_img = torch.from_numpy(src_img).to(torch.float32).unsqueeze(0)
        dst_img = torch.from_numpy(dst_img).to(torch.float32).unsqueeze(0)

        # src_img = (src_img - src_img.mean()) / src_img.std()
        # dst_img = (dst_img - dst_img.mean()) / dst_img.std()
        src_img = src_img / 255 + 1e-7
        dst_img = dst_img / 255 + 1e-7

        if self.pre_processing:
            src_dst_img = self.crop(torch.cat((src_img, dst_img), dim=0))
            src_img = src_dst_img[0].unsqueeze(0).unsqueeze(0)
            dst_img = src_dst_img[1].unsqueeze(0).unsqueeze(0)

        # save_png(src_img, "/scratch/conditional-flow-matching/src.png")
        # save_png(dst_img, "/scratch/conditional-flow-matching/dst.png")

        return tuple((src_img, dst_img))

class CFM_validation_jpeg(torch.utils.data.Dataset):
    def __init__(self, list_IDs, name=None, pre_processing=None):
        self.list_IDs = list_IDs
        self.name = name
        self.pre_processing = pre_processing

        self.crop = T.Compose([
            T.RandomCrop(256),
        ])

    def __len__(self):
        return len(self.list_IDs['high_dose'])

    def __getitem__(self, index):
        key = random.choice(['low_dose', 'mid_dose'])
        # abs_idx = index % len(self.list_IDs['high_dose'])

        src_items = list(self.list_IDs[key].items())
        dst_items = list(self.list_IDs['high_dose'].items())

        src_path, src_dose = src_items[index]
        dst_path, dst_dose = dst_items[index]

        src_img = load_dicom(src_path)
        dst_img = load_dicom(dst_path)

        src_img = dicom_to_8bit(src_img)
        dst_img = dicom_to_8bit(dst_img)

        src_img = torch.from_numpy(src_img).to(torch.float32).unsqueeze(0)
        dst_img = torch.from_numpy(dst_img).to(torch.float32).unsqueeze(0)

        # src_img = (src_img - src_img.mean()) / src_img.std()
        # dst_img = (dst_img - dst_img.mean()) / dst_img.std()
        src_img = src_img / 255 + 1e-7
        dst_img = dst_img / 255 + 1e-7

        if self.pre_processing:
            src_dst_img = self.crop(torch.cat((src_img, dst_img),dim=0))
            src_img = src_dst_img[0].unsqueeze(0).unsqueeze(0)
            dst_img = src_dst_img[1].unsqueeze(0).unsqueeze(0)

        # save_png(src_img, "/scratch/conditional-flow-matching/src.png")
        # save_png(dst_img, "/scratch/conditional-flow-matching/dst.png")

        return tuple((src_img, dst_img))