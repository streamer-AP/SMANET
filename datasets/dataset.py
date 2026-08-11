import os
import os.path as osp
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image

def canonical_dataset_name(datasetname):
    name = str(datasetname).strip()
    lower = name.lower()
    if lower in {"movingdronecrowd", "mdc"}:
        return "MovingDroneCrowd"
    if lower in {"sense", "sensecrowd"}:
        return "SENSE"
    if lower in {"crohd", "ht21"}:
        return "CroHD"
    raise NotImplementedError(f"Unsupported dataset: {datasetname}")

def _numeric_stem(path_or_name):
    return int(Path(path_or_name).stem)

def _image_files(directory):
    files = [
        name
        for name in os.listdir(directory)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return sorted(files, key=_numeric_stem)

def _open_rgb(path):
    return Image.open(path).convert("RGB")

def _empty_target(scene_name, frame_idx):
    return {
        "scene_name": scene_name,
        "frame": frame_idx,
        "person_id": torch.empty((0,), dtype=torch.long),
        "points": torch.empty((0, 2), dtype=torch.float32),
        "sigma": torch.empty((0,), dtype=torch.float32),
    }

def _expand_mdc_scenes(base_path, scene_names):
    expanded = []
    for scene_name in scene_names:
        scene_name = scene_name.strip()
        if not scene_name:
            continue
        if "/" in scene_name:
            expanded.append(scene_name)
            continue

        root = osp.join(base_path, "frames", scene_name)
        clip_names = [name for name in os.listdir(root) if "." not in name]
        for clip_name in sorted(clip_names):
            expanded.append(f"{scene_name}/{clip_name}")
    return expanded

def _normalize_sense_scenes(scene_names):
    return [line.strip().split()[0] for line in scene_names if line.strip()]

def _resolve_crohd_scene(base_path, scene_name, default_split="train"):
    scene_name = scene_name.strip().strip("/")
    if not scene_name:
        return scene_name
    if "/" in scene_name:
        if osp.isdir(osp.join(base_path, scene_name)):
            return scene_name
        split_name, bare_name = scene_name.rsplit("/", 1)
        split_root = osp.join(base_path, split_name)
        if osp.isdir(split_root):
            for candidate in sorted(os.listdir(split_root)):
                if candidate == bare_name or candidate.startswith(f"{bare_name}-"):
                    return osp.join(split_name, candidate)
        return scene_name
    for candidate in (
        scene_name,
        osp.join(default_split, scene_name),
        osp.join("train", scene_name),
        osp.join("val", scene_name),
        osp.join("test", scene_name),
    ):
        if osp.isdir(osp.join(base_path, candidate)):
            return candidate
    split_root = osp.join(base_path, default_split)
    if osp.isdir(split_root):
        for candidate in sorted(os.listdir(split_root)):
            if candidate == scene_name or candidate.startswith(f"{scene_name}-"):
                return osp.join(default_split, candidate)
    return osp.join(default_split, scene_name)

def _crohd_img_root(base_path, scene_name):
    scene_root = osp.join(base_path, scene_name)
    img1_root = osp.join(scene_root, "img1")
    return img1_root if osp.isdir(img1_root) else scene_root

def _crohd_gt_path(base_path, scene_name):
    scene_root = osp.join(base_path, scene_name)
    return osp.join(scene_root, "gt", "gt.txt")

def _has_crohd_target(base_path, scene_name):
    return osp.exists(_crohd_gt_path(base_path, scene_name))

def _has_sense_target(base_path, scene_name):
    return osp.exists(osp.join(base_path, "annotations", f"{scene_name}.txt"))

def _resolve_flow_dir(datasetname, base_path, image_path, flow_root_override=None):
    datasetname = canonical_dataset_name(datasetname)

    if datasetname == "MovingDroneCrowd":
        flow_root = flow_root_override or os.environ.get("MDC_FLOW_ROOT") or osp.join(base_path, "raft")
        parts = Path(image_path).parts
        return osp.join(flow_root, parts[-3], parts[-2])

    if datasetname == "SENSE":
        flow_root = flow_root_override or os.environ.get("SENSE_FLOW_ROOT") or osp.join(base_path, "raft")
        scene_name = Path(image_path).parts[-2]
        return osp.join(flow_root, scene_name)

    flow_root = (
        flow_root_override
        or os.environ.get("CROHD_FLOW_ROOT")
        or os.environ.get("HT21_FLOW_ROOT")
        or osp.join(base_path, "raft")
    )
    parts = Path(image_path).parts
    scene_name = parts[-3] if parts[-2] == "img1" else parts[-2]
    return osp.join(flow_root, scene_name)

def _load_flow_pair(datasetname, base_path, img_path_a, img_path_b, flow_root_override=None):
    flow_dir = _resolve_flow_dir(datasetname, base_path, img_path_a, flow_root_override)
    frame_a = _numeric_stem(img_path_a)
    frame_b = _numeric_stem(img_path_b)
    forward_flow_path = osp.join(flow_dir, f"{frame_a}_to_{frame_b}.pt")
    backward_flow_path = osp.join(flow_dir, f"{frame_b}_to_{frame_a}.pt")

    try:
        forward_flow = torch.load(forward_flow_path, map_location="cpu")
        backward_flow = torch.load(backward_flow_path, map_location="cpu")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Optical-flow file not found. Run the corresponding dataset preparation "
            f"script before loading the dataset. Missing {forward_flow_path} or "
            f"{backward_flow_path}."
        ) from exc

    return forward_flow.squeeze(0), backward_flow.squeeze(0)

def _build_masks(target0, target1):
    if "person_id" in target0 and "person_id" in target1:
        ids0 = target0["person_id"]
        ids1 = target1["person_id"]
        share_mask0 = torch.isin(ids0, ids1)
        share_mask1 = torch.isin(ids1, ids0)
        outflow_mask = torch.logical_not(share_mask0)
        inflow_mask = torch.logical_not(share_mask1)
        return share_mask0, share_mask1, outflow_mask, inflow_mask

    if "outflow" in target0 and "inflow" in target1:
        outflow_mask = target0["outflow"].bool()
        inflow_mask = target1["inflow"].bool()
        share_mask0 = torch.logical_not(outflow_mask)
        share_mask1 = torch.logical_not(inflow_mask)
        return share_mask0, share_mask1, outflow_mask, inflow_mask

    raise KeyError("Target dictionaries must contain person_id or inflow/outflow masks.")

def MDC_ImgPath_and_Target(base_path, scene_name):
    img_path = []
    labels = []
    root = osp.join(base_path, "frames", scene_name)
    img_ids = _image_files(root)

    annotation_path = root.replace("frames", "annotations") + ".csv"
    df = pd.read_csv(annotation_path, header=None)
    grouped = df.groupby(df[0])
    gts = {int(frame_id): group for frame_id, group in grouped}

    for img_id in img_ids:
        single_path = osp.join(root, img_id)
        frame_idx = _numeric_stem(img_id)
        if frame_idx - 1 in gts:
            label = gts[frame_idx - 1]
            data_tensor = torch.tensor(label.to_numpy())[:, 1:6].contiguous()
        else:
            data_tensor = torch.empty((0, 5))

        boxes = data_tensor[:, 1:5].float()
        points = torch.empty((len(boxes), 2), dtype=torch.float32)
        if len(boxes) > 0:
            points[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
            points[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
        person_ids = data_tensor[:, 0].long()

        img_path.append(single_path)
        labels.append(
            {
                "scene_name": scene_name,
                "frame": frame_idx,
                "person_id": person_ids,
                "points": points,
            }
        )
    return img_path, labels

def MDC_ImgPath(base_path, scene_name):
    root = osp.join(base_path, "frames", scene_name)
    return [osp.join(root, img_id) for img_id in _image_files(root)]

def SENSE_ImgPath_and_Target(base_path, scene_name):
    img_path = []
    labels = []
    root = osp.join(base_path, "videos", scene_name)
    img_ids = _image_files(root)
    annotation_path = osp.join(base_path, "annotations", f"{scene_name}.txt")

    gts = defaultdict(list)
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()
            if not fields:
                continue
            values = [float(value) for value in fields[3:]]
            if len(values) % 7 != 0:
                raise ValueError(f"Invalid SenseCrowd annotation line for {fields[0]}")
            gts[fields[0]] = values

    for img_id in img_ids:
        single_path = osp.join(root, img_id)
        frame_idx = _numeric_stem(img_id)
        raw = gts.get(img_id, [])
        if raw:
            box_and_point = torch.tensor(raw, dtype=torch.float32).view(-1, 7).contiguous()
            points = box_and_point[:, 4:6].float()
            person_ids = box_and_point[:, 6].long()
            sigma = 0.6 * torch.stack(
                [
                    (box_and_point[:, 2] - box_and_point[:, 0]) / 2,
                    (box_and_point[:, 3] - box_and_point[:, 1]) / 2,
                ],
                dim=1,
            ).min(dim=1)[0]
        else:
            points = torch.empty((0, 2), dtype=torch.float32)
            person_ids = torch.empty((0,), dtype=torch.long)
            sigma = torch.empty((0,), dtype=torch.float32)

        img_path.append(single_path)
        labels.append(
            {
                "scene_name": scene_name,
                "frame": frame_idx,
                "person_id": person_ids,
                "points": points,
                "sigma": sigma,
            }
        )
    return img_path, labels

def SENSE_ImgPath(base_path, scene_name):
    root = osp.join(base_path, "videos", scene_name)
    return [osp.join(root, img_id) for img_id in _image_files(root)]

def CroHD_ImgPath_and_Target(base_path, scene_name):
    img_path = []
    labels = []
    root = _crohd_img_root(base_path, scene_name)
    img_ids = _image_files(root)
    gt_path = _crohd_gt_path(base_path, scene_name)

    gts = defaultdict(list)
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            fields = [float(value) for value in line.strip().split(",") if value != ""]
            if len(fields) < 6:
                continue
            gts[int(fields[0])].append(fields)

    for img_id in img_ids:
        single_path = osp.join(root, img_id)
        frame_idx = _numeric_stem(img_id)
        annotation = gts.get(frame_idx, [])
        if annotation:
            data_tensor = torch.tensor(annotation, dtype=torch.float32)
            boxes = data_tensor[:, 2:6]
            points = boxes[:, 0:2] + boxes[:, 2:4] / 2
            person_ids = data_tensor[:, 1].long()
            sigma = torch.min(boxes[:, 2:4], dim=1)[0] / 2
            target = {
                "scene_name": scene_name,
                "frame": frame_idx,
                "person_id": person_ids,
                "points": points,
                "sigma": sigma,
            }
        else:
            target = _empty_target(scene_name, frame_idx)

        img_path.append(single_path)
        labels.append(target)
    return img_path, labels

def CroHD_ImgPath(base_path, scene_name):
    root = _crohd_img_root(base_path, scene_name)
    return [osp.join(root, img_id) for img_id in _image_files(root)]

class Dataset(data.Dataset):
    def __init__(
        self,
        txt_path,
        base_path,
        main_transform=None,
        img_transform=None,
        train=True,
        datasetname="MovingDroneCrowd",
        frame_intervals=(3, 6),
        flow_root=None,
        min_shared_people=3,
    ):
        self.base_path = base_path
        self.imgs_path = []
        self.labels = []
        self.datasetname = canonical_dataset_name(datasetname)
        self.is_train = train
        self.main_transforms = main_transform
        self.img_transforms = img_transform
        self.frame_intervals = frame_intervals
        self.flow_root = flow_root
        self.min_shared_people = min_shared_people

        with open(osp.join(base_path, txt_path), "r", encoding="utf-8") as txt:
            scene_names = [line.strip() for line in txt if line.strip()]

        if self.datasetname == "MovingDroneCrowd":
            scene_names = _expand_mdc_scenes(base_path, scene_names)
        elif self.datasetname == "SENSE":
            scene_names = _normalize_sense_scenes(scene_names)
        elif self.datasetname == "CroHD":
            scene_names = [_resolve_crohd_scene(base_path, scene, "train") for scene in scene_names]

        for scene_name in scene_names:
            if self.datasetname == "MovingDroneCrowd":
                img_path, label = MDC_ImgPath_and_Target(base_path, scene_name)
            elif self.datasetname == "SENSE":
                img_path, label = SENSE_ImgPath_and_Target(base_path, scene_name)
            else:
                img_path, label = CroHD_ImgPath_and_Target(base_path, scene_name)

            if train and len(img_path) // 2 < frame_intervals[0]:
                print(
                    f"Warning: skip short scene {scene_name}: "
                    f"frames={len(img_path)}, min_interval={frame_intervals[0]}"
                )
                continue
            self.imgs_path += img_path
            self.labels += label

        self.scenes = []
        self.scene_id = defaultdict(int)
        for label in self.labels:
            scene_name = label["scene_name"]
            self.scene_id[scene_name] += 1
            self.scenes.append(scene_name)

        self.n_sample = len(self.imgs_path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        c = index
        scene_name = self.scenes[c]
        max_interval = min(self.scene_id[scene_name] // 2, self.frame_intervals[1])
        if max_interval < self.frame_intervals[0]:
            return self.__getitem__((index + 1) % len(self))

        tmp_intervals = random.randint(self.frame_intervals[0], max_interval)
        if c < self.n_sample - tmp_intervals and self.scenes[c + tmp_intervals] == scene_name:
            pair_c = c + tmp_intervals
        else:
            pair_c = c
            c = c - tmp_intervals
        assert self.scenes[c] == self.scenes[pair_c]

        img0 = _open_rgb(self.imgs_path[c])
        img1 = _open_rgb(self.imgs_path[pair_c])
        target0 = deepcopy(self.labels[c])
        target1 = deepcopy(self.labels[pair_c])

        forward_flow, backward_flow = _load_flow_pair(
            self.datasetname,
            self.base_path,
            self.imgs_path[c],
            self.imgs_path[pair_c],
            self.flow_root,
        )

        if self.main_transforms is not None:
            if not hasattr(self.main_transforms, "__call_pair__"):
                raise TypeError("Training spatial transforms must implement __call_pair__ for flow alignment")
            (
                img0,
                target0,
                img1,
                target1,
                forward_flow,
                backward_flow,
            ) = self.main_transforms.__call_pair__(
                img0,
                target0,
                img1,
                target1,
                forward_flow,
                backward_flow,
            )

        share_mask0, share_mask1, outflow_mask, inflow_mask = _build_masks(target0, target1)
        count_in_pair = [target0["points"].size(0), target1["points"].size(0)]
        if not (
            all(count > 0 for count in count_in_pair)
            and torch.sum(share_mask0) >= self.min_shared_people
        ):
            return self.__getitem__((index + 1) % len(self))

        target0["share_mask0"] = share_mask0
        target0["outflow_mask"] = outflow_mask
        target1["share_mask1"] = share_mask1
        target1["inflow_mask"] = inflow_mask

        if self.img_transforms is not None:
            img0 = self.img_transforms(img0)
            img1 = self.img_transforms(img1)

        return [img0, img1], [target0, target1], [forward_flow, backward_flow]

class TestDataset(data.Dataset):
    def __init__(
        self,
        scene_name,
        base_path,
        main_transform=None,
        img_transform=None,
        interval=1,
        skip_flag=True,
        target=True,
        datasetname="MovingDroneCrowd",
        flow_root=None,
    ):
        self.base_path = base_path
        self.target = target
        self.datasetname = canonical_dataset_name(datasetname)
        self.main_transforms = main_transform
        self.img_transforms = img_transform
        self.flow_root = flow_root

        if self.datasetname == "CroHD":
            scene_name = _resolve_crohd_scene(base_path, scene_name, "test")
            if self.target and not _has_crohd_target(base_path, scene_name):
                self.target = False
        elif self.datasetname == "SENSE":
            scene_name = scene_name.strip().split()[0]
            if self.target and not _has_sense_target(base_path, scene_name):
                self.target = False

        if self.target:
            if self.datasetname == "MovingDroneCrowd":
                self.imgs_path, self.label = MDC_ImgPath_and_Target(self.base_path, scene_name)
            elif self.datasetname == "SENSE":
                self.imgs_path, self.label = SENSE_ImgPath_and_Target(self.base_path, scene_name)
            else:
                self.imgs_path, self.label = CroHD_ImgPath_and_Target(self.base_path, scene_name)
        else:
            if self.datasetname == "MovingDroneCrowd":
                self.imgs_path = MDC_ImgPath(self.base_path, scene_name)
            elif self.datasetname == "SENSE":
                self.imgs_path = SENSE_ImgPath(self.base_path, scene_name)
            else:
                self.imgs_path = CroHD_ImgPath(self.base_path, scene_name)
            self.label = None

        self.length = len(self.imgs_path)
        self.interval = interval if 0 < interval < self.length else max(1, self.length // 2)
        self.skip_flag = skip_flag
        self.valid = self.is_valid()

    def is_valid(self):
        if not self.skip_flag:
            return torch.ones((self.length))

        valid = torch.zeros((self.length))
        loop_idx_range = self.length - self.interval - 1
        for i in range(max(0, self.length - self.interval)):
            if i % self.interval == 0:
                valid[i] = 1
                if i + self.interval > loop_idx_range and i + self.interval < self.length:
                    valid[i + self.interval] = 1
            elif i == loop_idx_range:
                valid[i] = 1
                if i + self.interval < self.length:
                    valid[i + self.interval] = 1
        return valid

    def __len__(self):
        return max(0, self.length - self.interval)

    def __getitem__(self, index):
        if not self.valid[index]:
            return None, None, None

        index1 = index
        index2 = index + self.interval
        img1 = _open_rgb(self.imgs_path[index1])
        img2 = _open_rgb(self.imgs_path[index2])

        if self.target:
            target1 = deepcopy(self.label[index1])
            target2 = deepcopy(self.label[index2])

            if self.main_transforms is not None:
                img1, target1 = self.main_transforms(img1, target1)
                img2, target2 = self.main_transforms(img2, target2)

            share_mask0, share_mask1, outflow_mask, inflow_mask = _build_masks(target1, target2)
            target1["share_mask0"] = share_mask0
            target1["outflow_mask"] = outflow_mask
            target2["share_mask1"] = share_mask1
            target2["inflow_mask"] = inflow_mask

            if self.img_transforms is not None:
                img1 = self.img_transforms(img1)
                img2 = self.img_transforms(img2)

            forward_flow, backward_flow = _load_flow_pair(
                self.datasetname,
                self.base_path,
                self.imgs_path[index1],
                self.imgs_path[index2],
                self.flow_root,
            )
            return [img1, img2], [target1, target2], [forward_flow, backward_flow]

        if self.img_transforms is not None:
            img1 = self.img_transforms(img1)
            img2 = self.img_transforms(img2)

        forward_flow, backward_flow = _load_flow_pair(
            self.datasetname,
            self.base_path,
            self.imgs_path[index1],
            self.imgs_path[index2],
            self.flow_root,
        )
        return [img1, img2], None, [forward_flow, backward_flow]
