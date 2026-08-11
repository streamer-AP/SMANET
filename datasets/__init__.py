import os
import random

import torch
import torchvision.transforms as standard_transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import misc.transforms as own_transforms
from config import cfg
from misc.transforms import check_image

from . import dataset
from . import setting

def _setting_key(datasetname):
    return dataset.canonical_dataset_name(datasetname)

def _split_file(cfg_data, mode):
    list_name = f"{mode.upper()}_LST"
    return os.path.join(cfg_data.DATA_PATH, getattr(cfg_data, list_name))

def _read_scene_names(datasetname, cfg_data, mode):
    split_path = _split_file(cfg_data, mode)
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as txt:
            return [line.strip() for line in txt if line.strip()]

    if datasetname == "MovingDroneCrowd":
        frame_root = os.path.join(cfg_data.DATA_PATH, "frames")
        if os.path.isdir(frame_root):
            return sorted(name for name in os.listdir(frame_root) if os.path.isdir(os.path.join(frame_root, name)))

    if datasetname == "SENSE":
        video_root = os.path.join(cfg_data.DATA_PATH, "videos")
        if os.path.isdir(video_root):
            return sorted(name for name in os.listdir(video_root) if os.path.isdir(os.path.join(video_root, name)))

    if datasetname == "CroHD":
        mode_root = os.path.join(cfg_data.DATA_PATH, mode)
        if os.path.isdir(mode_root):
            return [
                os.path.join(mode, name)
                for name in sorted(os.listdir(mode_root))
                if os.path.isdir(os.path.join(mode_root, name))
            ]

    raise FileNotFoundError(f"Split file not found: {split_path}")

def _normalize_eval_scenes(datasetname, cfg_data, scene_names, mode):
    if datasetname == "MovingDroneCrowd":
        return dataset._expand_mdc_scenes(cfg_data.DATA_PATH, scene_names)
    if datasetname == "SENSE":
        return [line.split()[0] for line in scene_names if line.strip()]
    if datasetname == "CroHD":
        return [
            dataset._resolve_crohd_scene(cfg_data.DATA_PATH, scene_name, mode)
            for scene_name in scene_names
            if scene_name.strip()
        ]
    return scene_names

class train_pair_transform(object):
    def __init__(self, cfg_data, check_dim=True):
        self.cfg_data = cfg_data
        self.pair_flag = 0
        self.scale_factor = 1
        self.last_cw_ch = (0, 0)
        self.crop_left = (0, 0)
        self.last_crop_left = (0, 0)
        self.rate_range = (cfg.TRAIN_SCALE_MIN, cfg.TRAIN_SCALE_MAX)
        self.resize_and_crop = own_transforms.RandomCrop()
        self.scale_to_setting = own_transforms.ScaleByRateWithMin(cfg_data.TRAIN_SIZE[1], cfg_data.TRAIN_SIZE[0])
        self.flip_flag = 0
        self.horizontal_flip = own_transforms.RandomHorizontallyFlip()
        self.last_frame_size = (0, 0)
        self.check_dim = check_dim

    def _prepare(self, img, target, flow):
        flow_size = tuple(flow.shape[-2:])
        flow = own_transforms.flow_to_image_size(
            flow,
            (img.size[1], img.size[0]),
            self.cfg_data.TRAIN_SIZE,
        )
        img, target, flow = check_image(
            img,
            target,
            (self.c_h, self.c_w),
            (self.cfg_data.TRAINING_MAX_LONG, self.cfg_data.TRAINING_MAX_SHORT),
            flow=flow,
        )
        return img, target, flow, flow_size

    def _finish(self, img, target, flow, crop_left, flow_size, flip_flag=0):
        img, target, flow = self.resize_and_crop(
            img,
            target,
            crop_left,
            crop_size=(self.c_h, self.c_w),
            flow=flow,
        )
        img, target, flow = self.scale_to_setting(img, target, flow=flow)

        target["points"][:, 0] = torch.clamp(target["points"][:, 0], min=0, max=img.size[0] - 1)
        target["points"][:, 1] = torch.clamp(target["points"][:, 1], min=0, max=img.size[1] - 1)
        img, target, flow = self.horizontal_flip(
            img,
            target,
            flip_flag,
            flow=flow,
        )

        flow = own_transforms.resize_flow(
            flow,
            flow_size,
            flow_size[1] / img.size[0],
            flow_size[0] / img.size[1],
        )
        return img, target, flow

    def __call_pair__(self, img0, target0, img1, target1, forward_flow, backward_flow):
        self.scale_factor = random.uniform(self.rate_range[0], self.rate_range[1])
        self.c_h = int(self.cfg_data.TRAIN_SIZE[0] / self.scale_factor)
        self.c_w = int(self.cfg_data.TRAIN_SIZE[1] / self.scale_factor)
        img0, target0, forward_flow, forward_size = self._prepare(img0, target0, forward_flow)
        img1, target1, backward_flow, backward_size = self._prepare(img1, target1, backward_flow)

        w0, h0 = img0.size
        x1 = random.randint(0, max(0, w0 - self.c_w))
        y1 = random.randint(0, max(0, h0 - self.c_h))
        crop0 = (x1, y1)

        w1, h1 = img1.size
        crop1 = (
            min(x1, max(0, w1 - self.c_w)),
            min(y1, max(0, h1 - self.c_h)),
        )
        flip_flag = 0

        img0, target0, forward_flow = self._finish(
            img0, target0, forward_flow, crop0, forward_size, flip_flag
        )
        img1, target1, backward_flow = self._finish(
            img1, target1, backward_flow, crop1, backward_size, flip_flag
        )
        return img0, target0, img1, target1, forward_flow, backward_flow

    def __call__(self, img, target, flow=None):
        if flow is None:
            raise ValueError("train_pair_transform requires flow for spatially aligned training")

        self.scale_factor = random.uniform(self.rate_range[0], self.rate_range[1])
        self.c_h = int(self.cfg_data.TRAIN_SIZE[0] / self.scale_factor)
        self.c_w = int(self.cfg_data.TRAIN_SIZE[1] / self.scale_factor)
        img, target, flow, flow_size = self._prepare(img, target, flow)
        w, h = img.size
        crop_left = (
            random.randint(0, max(0, w - self.c_w)),
            random.randint(0, max(0, h - self.c_h)),
        )
        return self._finish(img, target, flow, crop_left, flow_size)

class test_transform(object):
    def __init__(self, cfg_data):
        self.cfg_data = cfg_data

    def __call__(self, img, target):
        w, h = img.size
        long_side = max(w, h)
        short_side = min(w, h)
        if self.cfg_data.TEST_MAX_LONG is not None and self.cfg_data.TEST_MAX_SHORT is not None:
            max_long_side = self.cfg_data.TEST_MAX_LONG
            max_short_side = self.cfg_data.TEST_MAX_SHORT
            scale_long = max_long_side / long_side
            scale_short = max_short_side / short_side
            if scale_long < 1 or scale_short < 1:
                scale = min(scale_long, scale_short)
                new_width = int(w * scale)
                new_height = int(h * scale)
                target["points"] = target["points"] * scale
                img = img.resize((new_width, new_height), Image.LANCZOS)

        target["points"][:, 0] = torch.clamp(target["points"][:, 0], min=0, max=img.size[0] - 1)
        target["points"][:, 1] = torch.clamp(target["points"][:, 1], min=0, max=img.size[1] - 1)
        return img, target

class train_resize_transform(object):
    def __init__(self, new_h, new_w):
        self.new_h = new_h
        self.new_w = new_w
        self.horizontal_flip = own_transforms.RandomHorizontallyFlip()

    def __call__(self, img, target):
        w, h = img.size
        img = img.resize((self.new_w, self.new_h), Image.LANCZOS)
        rate_w = self.new_w / w
        rate_h = self.new_h / h
        target["points"][:, 0] = target["points"][:, 0] * rate_w
        target["points"][:, 1] = target["points"][:, 1] * rate_h
        target["points"][:, 0] = torch.clamp(target["points"][:, 0], min=0, max=self.new_w - 1)
        target["points"][:, 1] = torch.clamp(target["points"][:, 1], min=0, max=self.new_h - 1)
        img, target = self.horizontal_flip(img, target, 0)
        return img, target

def collate_fn(batch):
    batch = [item for item in batch if item is not None and item[0] is not None]
    if not batch:
        return None, None, None

    img_pairs, label_pairs, flow_pairs = [], [], []
    for item in batch:
        img_pairs.append(item[0])
        if len(item) > 1 and item[1] is not None:
            label_pairs.append(item[1])
        if len(item) > 2 and item[2] is not None:
            flow_pairs.append(item[2])

    img_tensors = []
    valid_img_pairs = [pair for pair in img_pairs if pair is not None and all(img is not None for img in pair)]
    if valid_img_pairs:
        try:
            img_tensors = torch.cat([torch.stack(pair, dim=0) for pair in valid_img_pairs], dim=0)
        except Exception:
            img_tensors = []

    labels = []
    for label_pair in label_pairs:
        labels.extend(label for label in label_pair if label is not None)

    flows = []
    forward_flows = []
    backward_flows = []
    for flow_pair in flow_pairs:
        if flow_pair is not None and len(flow_pair) == 2 and flow_pair[0] is not None and flow_pair[1] is not None:
            forward_flows.append(flow_pair[0])
            backward_flows.append(flow_pair[1])
    if forward_flows and backward_flows:
        try:
            flows = [torch.stack(forward_flows, dim=0), torch.stack(backward_flows, dim=0)]
        except Exception:
            flows = []

    if not torch.is_tensor(img_tensors) or img_tensors.nelement() == 0:
        return None, None, None

    return img_tensors, labels, flows

def createTrainData(datasetname, Dataset, cfg_data, distributed):
    img_transform = standard_transforms.Compose(
        [
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*cfg_data.MEAN_STD),
        ]
    )
    pair_transform = train_pair_transform(cfg_data)
    train_set = Dataset(
        cfg_data.TRAIN_LST,
        cfg_data.DATA_PATH,
        main_transform=pair_transform,
        img_transform=img_transform,
        train=True,
        datasetname=datasetname,
        frame_intervals=cfg_data.TRAIN_FRAME_INTERVALS,
        flow_root=getattr(cfg_data, "FLOW_ROOT", None),
        min_shared_people=cfg.MIN_SHARED_PEOPLE,
    )
    sampler_train = DistributedSampler(train_set) if distributed else None
    train_num_workers = getattr(cfg, "TRAIN_NUM_WORKERS", 32)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg_data.TRAIN_BATCH_SIZE,
        sampler=sampler_train,
        shuffle=(sampler_train is None),
        num_workers=train_num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print("dataset is {}, training images num is {}".format(datasetname, train_set.__len__()))
    return train_loader, sampler_train

def createValData(datasetname, Dataset, cfg_data):
    img_transform = standard_transforms.Compose(
        [
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*cfg_data.MEAN_STD),
        ]
    )
    val_loader = []
    with open(os.path.join(cfg_data.DATA_PATH, cfg_data.VAL_LST), "r", encoding="utf-8") as txt:
        scene_names = [line.strip() for line in txt if line.strip()]
    for scene in scene_names:
        sub_val_dataset = Dataset(
            [scene],
            cfg_data.DATA_PATH,
            main_transform=None,
            img_transform=img_transform,
            train=False,
            datasetname=datasetname,
            flow_root=getattr(cfg_data, "FLOW_ROOT", None),
        )
        val_num_workers = getattr(cfg, "VAL_NUM_WORKERS", 4)
        sub_val_loader = DataLoader(
            sub_val_dataset,
            batch_size=cfg_data.VAL_BATCH_SIZE,
            num_workers=val_num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
        )
        val_loader.append(sub_val_loader)
    return val_loader

def createRestore(mean_std):
    return standard_transforms.Compose(
        [
            own_transforms.DeNormalize(*mean_std),
            standard_transforms.ToPILImage(),
        ]
    )

def loading_data(datasetname, val_interval, distributed, is_main):
    datasetname = _setting_key(datasetname)
    cfg_data = getattr(setting, datasetname).cfg_data

    train_loader, sampler_train = createTrainData(datasetname, dataset.Dataset, cfg_data, distributed)
    restore_transform = createRestore(cfg_data.MEAN_STD)
    val_loader = createValTestData(
        datasetname,
        dataset.TestDataset,
        cfg_data,
        val_interval,
        True,
        is_main,
        mode="val",
        flow_root=getattr(cfg_data, "FLOW_ROOT", None),
        num_workers=getattr(cfg, "VAL_NUM_WORKERS", 8),
    )

    return train_loader, sampler_train, val_loader, restore_transform

def createValTestData(
    datasetname,
    Dataset,
    cfg_data,
    frame_interval,
    skip_flag,
    is_main,
    mode="val",
    flow_root=None,
    num_workers=8,
    scene_shard_id=None,
    scene_shards=1,
):
    datasetname = _setting_key(datasetname)
    img_transform = standard_transforms.Compose(
        [
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*cfg_data.MEAN_STD),
        ]
    )
    main_transform = test_transform(cfg_data)
    target = True

    scene_names = _read_scene_names(datasetname, cfg_data, mode)
    scene_names = _normalize_eval_scenes(datasetname, cfg_data, scene_names, mode)

    if scene_shard_id is not None and scene_shards > 1:
        total_scenes = len(scene_names)
        scene_names = [
            scene_name
            for idx, scene_name in enumerate(scene_names)
            if idx % scene_shards == scene_shard_id
        ]
        print(f"[SceneShard] {scene_shard_id}/{scene_shards}: {len(scene_names)}/{total_scenes} scenes")

    data_loader = []
    for scene_name in scene_names:
        print(scene_name)
        sub_dataset = Dataset(
            scene_name=scene_name,
            base_path=cfg_data.DATA_PATH,
            main_transform=main_transform,
            img_transform=img_transform,
            interval=frame_interval,
            skip_flag=skip_flag,
            target=target,
            datasetname=datasetname,
            flow_root=flow_root or getattr(cfg_data, "FLOW_ROOT", None),
        )
        sub_loader = DataLoader(
            sub_dataset,
            batch_size=cfg_data.VAL_BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=(num_workers > 0),
        )
        data_loader.append([scene_name, sub_loader])
    return data_loader

def loading_testset(
    datasetname,
    test_interval,
    skip_flag,
    mode="test",
    flow_root=None,
    num_workers=0,
    scene_shard_id=None,
    scene_shards=1,
):
    datasetname = _setting_key(datasetname)
    cfg_data = getattr(setting, datasetname).cfg_data
    test_loader = createValTestData(
        datasetname,
        dataset.TestDataset,
        cfg_data,
        test_interval,
        skip_flag,
        True,
        mode=mode,
        flow_root=flow_root or getattr(cfg_data, "FLOW_ROOT", None),
        num_workers=num_workers,
        scene_shard_id=scene_shard_id,
        scene_shards=scene_shards,
    )
    restore_transform = createRestore(cfg_data.MEAN_STD)
    return test_loader, restore_transform
