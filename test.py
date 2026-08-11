import argparse
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

import datasets
from config import cfg
from datasets import dataset as dataset_lib
from misc.utils import AverageMeter, compute_metrics_single_scene, save_test_visual
from model.smanet import SMANet, remap_legacy_state_dict

HT21_GT_COUNTS = {
    "HT21-11": 133,
    "HT21-12": 737,
    "HT21-13": 734,
    "HT21-14": 1040,
    "HT21-15": 321,
}

parser = argparse.ArgumentParser(
    description="VIC test and demo",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--DATASET", type=str, default="MovingDroneCrowd", help="Dataset name used for test")
parser.add_argument("--data_path", type=str, default="", help="Override dataset root path")
parser.add_argument("--flow_root", type=str, default="", help="Override optical-flow root path")
parser.add_argument("--output_dir", type=str, default="test_results", help="Directory where to write test results")
parser.add_argument("--test_name", type=str, default="check", help="Test name used to identify different tests")
parser.add_argument("--test_split", type=str, choices=("val", "test"), default="test", help="Dataset split to evaluate")
parser.add_argument("--test_intervals", type=int, default=4, help="Frame interval for test")
parser.add_argument(
    "--skip_flag",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="skip unmatched frame pairs when evaluating scene-level counts",
)
parser.add_argument("--SEED", type=int, default=3035, help="seed")
parser.add_argument(
    "--model_path",
    type=str,
    default=os.environ.get("VIC_MODEL_PATH", "checkpoints/MDC_SMANet.pth"),
    help="pretrained weight path",
)
parser.add_argument("--GPU_ID", type=str, default="0", help="gpu id for test")
parser.add_argument("--save_visual", action=argparse.BooleanOptionalAction, default=True, help="save visual results")
parser.add_argument("--max_scenes", type=int, default=0, help="limit tested scenes for smoke tests")
parser.add_argument("--max_batches", type=int, default=0, help="limit batches per scene for smoke tests")

opt = parser.parse_args()
opt.output_dir = os.path.join(opt.output_dir, opt.DATASET, opt.test_name)
os.makedirs(opt.output_dir, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_ID

def module2model(module_state_dict):
    if isinstance(module_state_dict, dict) and "net" in module_state_dict:
        module_state_dict = module_state_dict["net"]

    state_dict = {}
    skipped = []
    for k, v in module_state_dict.items():
        while k.startswith("module."):
            k = k[7:]
        if k == "alpha" or k.endswith(".alpha"):
            skipped.append(k)
            continue
        state_dict[k] = v
    if skipped:
        print(f"[Info] Skip non-model checkpoint keys: {sorted(set(skipped))}")
    return state_dict

def _scene_name_key(scene_name):
    basename = Path(scene_name).name
    if basename.startswith("HT21-"):
        parts = basename.split("-")
        if len(parts) >= 2:
            return "-".join(parts[:2])
    return scene_name

def _load_label_map(data_path, dataset_name):
    label_files = []
    if dataset_name == "MovingDroneCrowd":
        label_files = ["scene_label.txt"]
    elif dataset_name == "SENSE":
        label_files = ["scene_labels.txt", "scene_label.txt"]

    for filename in label_files:
        path = os.path.join(data_path, filename)
        if not os.path.exists(path):
            continue
        labels = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    labels[parts[0]] = [int(x) for x in parts[1:]]
        if labels:
            print(f"[Info] Loaded {len(labels)} scene labels from {path}")
            return labels
    return {}

def _pred_scene_count(pred_dict, intervals):
    count = pred_dict["first_frame"]
    inflow = pred_dict["inflow"]
    for idx, value in enumerate(inflow):
        if idx % intervals == 0 or idx == len(inflow) - 1:
            count += value
    return count

def _compute_video_metrics(scenes_pred_dict, scenes_gt_dict, intervals):
    scene_cnt = len(scenes_pred_dict)
    counts = torch.zeros(scene_cnt, 2)
    wrae_parts = torch.zeros(scene_cnt, 2)

    for idx, (pred_dict, gt_dict) in enumerate(zip(scenes_pred_dict, scenes_gt_dict)):
        time = pred_dict["time"]
        pred_count, gt_count, _, _ = compute_metrics_single_scene(pred_dict, gt_dict, intervals)
        abs_err = abs(pred_count - gt_count)
        counts[idx, :] = torch.tensor([pred_count, gt_count])
        wrae_parts[idx, :] = torch.tensor([abs_err / (gt_count + 1e-10), time])

    mae = torch.mean(torch.abs(counts[:, 0] - counts[:, 1]))
    mse = torch.mean((counts[:, 0] - counts[:, 1]) ** 2).sqrt()
    weights = wrae_parts[:, 1] / (wrae_parts[:, 1].sum() + 1e-10)
    wrae = torch.sum(wrae_parts[:, 0] * weights) * 100.0

    return mae, mse, wrae, counts

def _write_density_results(save_dir, density_keys, scenes_pred_dict, scenes_gt_dict, intervals):
    result_file = os.path.join(save_dir, "results.txt")
    save_cnt_result = None
    with open(result_file, "w", encoding="utf-8") as f:
        for key in density_keys:
            s_pred_dict = scenes_pred_dict[key]
            s_gt_dict = scenes_gt_dict[key]
            if not s_pred_dict:
                continue
            mae, mse, wrae, cnt_result = _compute_video_metrics(s_pred_dict, s_gt_dict, intervals)
            if key == "all":
                save_cnt_result = cnt_result
            print("=" * 20, key, "=" * 20)
            print("MAE: %.2f, MSE: %.2f  WRAE: %.2f" % (mae.data, mse.data, wrae.data))
            f.write(f"{'=' * 20}{key}{'=' * 20}\n")
            f.write(f"MAE: {mae.item():.2f}, MSE: {mse.item():.2f}  WRAE: {wrae.item():.2f}\n")
        if save_cnt_result is not None:
            pre_vs_gt_msg = f"Pre vs GT: {save_cnt_result}"
            print(pre_vs_gt_msg)
            f.write(pre_vs_gt_msg + "\n")
    print(f"[Info] results saved: {result_file}")

def _write_scene_results(save_dir, scene_records):
    pred = torch.tensor([item["pred"] for item in scene_records], dtype=torch.float32)
    gt = torch.tensor([item["gt"] for item in scene_records], dtype=torch.float32)
    time = torch.tensor([item["time"] for item in scene_records], dtype=torch.float32)
    abs_err = torch.abs(pred - gt)
    mae = abs_err.mean()
    mse = torch.mean((pred - gt) ** 2).sqrt()
    wrae = torch.sum((abs_err / (gt + 1e-10)) * (time / (time.sum() + 1e-10))) * 100.0

    result_file = os.path.join(save_dir, "results.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        print("=" * 20, "all", "=" * 20)
        print("MAE: %.2f, MSE: %.2f, WRAE: %.2f" % (mae.item(), mse.item(), wrae.item()))
        f.write("============= all =============\n")
        f.write(f"MAE: {mae.item():.2f}, MSE: {mse.item():.2f}, WRAE: {wrae.item():.2f}\n")
        f.write("scene_name\tpred\tgt\tabs_err\n")
        for item in scene_records:
            f.write(f"{item['scene_name']}\t{item['pred']:.2f}\t{item['gt']:.2f}\t{item['abs_err']:.2f}\n")
    print(f"[Info] results saved: {result_file}")
    return mae, mse, wrae

def test(cfg_data):
    dataset_name = dataset_lib.canonical_dataset_name(opt.DATASET)
    cfg.DATASET = dataset_name
    if opt.data_path:
        cfg_data.DATA_PATH = opt.data_path
    if opt.flow_root:
        cfg_data.FLOW_ROOT = opt.flow_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SMANet(cfg, cfg_data)
    model.to(device)

    if not os.path.isfile(opt.model_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {opt.model_path}. Pass --model_path or set VIC_MODEL_PATH."
        )

    test_loader, restore_transform = datasets.loading_testset(
        opt.DATASET,
        opt.test_intervals,
        opt.skip_flag,
        mode=opt.test_split,
        flow_root=opt.flow_root or None,
    )

    state_dict = torch.load(opt.model_path, map_location="cpu")
    model.load_state_dict(remap_legacy_state_dict(module2model(state_dict)), strict=True)
    model.eval()

    sing_cnt_errors = {"mae": AverageMeter(), "mse": AverageMeter()}
    has_frame_gt = dataset_name in {"MovingDroneCrowd", "SENSE"}
    density_labels = _load_label_map(cfg_data.DATA_PATH, dataset_name) if has_frame_gt else {}

    if dataset_name == "MovingDroneCrowd":
        density_keys = ["all", "density0", "density1", "density2", "density3"]
        scenes_pred_dict = {k: [] for k in density_keys}
        scenes_gt_dict = {k: [] for k in density_keys}
    elif dataset_name == "SENSE":
        density_keys = ["all", "density0", "density1", "density2", "density3", "density4"]
        scenes_pred_dict = {k: [] for k in density_keys}
        scenes_gt_dict = {k: [] for k in density_keys}
    else:
        density_keys = []
        scene_records = []

    intervals = 1 if opt.skip_flag else opt.test_intervals

    for scene_id, (scene_name, sub_valset) in enumerate(test_loader, 0):
        if opt.max_scenes and scene_id >= opt.max_scenes:
            break
        scene_key = _scene_name_key(scene_name)
        gen_tqdm = tqdm(sub_valset, desc=scene_key)
        video_time = len(sub_valset) + opt.test_intervals
        pred_dict = {"id": scene_id, "time": video_time, "first_frame": 0, "inflow": [], "outflow": []}

        gt_dict = {"id": scene_id, "time": video_time, "first_frame": 0, "inflow": [], "outflow": []}
        scene_gt_count = HT21_GT_COUNTS.get(scene_key) if not has_frame_gt else None

        for vi, data in enumerate(gen_tqdm, 0):
            if opt.max_batches and vi >= opt.max_batches:
                break
            if data is None or data[0] is None:
                continue

            if vi % opt.test_intervals == 0 or vi == len(sub_valset) - 1:
                frame_signal = "match"
            else:
                frame_signal = "skip"

            if frame_signal != "match" and opt.skip_flag:
                continue

            img, label, flow = data
            if not label:
                label = None
            if label is not None:
                for i in range(len(label)):
                    for key, value in label[i].items():
                        if torch.is_tensor(value):
                            label[i][key] = value.to(device)

            img = img.to(device)
            flow = [f.to(device) for f in flow]

            with torch.no_grad():
                b, _, h, w = img.shape
                stride = cfg.INPUT_STRIDE
                pad_h = (stride - h % stride) % stride
                pad_w = (stride - w % stride) % stride
                img = F.pad(img, (0, pad_w, 0, pad_h), "constant")
                h, w = img.size(2), img.size(3)
                placeholder = torch.zeros((1, h, w), device=img.device)

                gt_global_den = None
                gt_in_out_den = None
                if label is not None:
                    gt_global_dot = torch.zeros((b, 1, h, w), device=img.device)
                    gt_in_out_dot = torch.zeros((b, 1, h, w), device=img.device)
                    gt_share_dot = torch.zeros((b, 1, h, w), device=img.device)

                    for i in range(b):
                        points = label[i]["points"].long()
                        if points.numel() > 0:
                            gt_global_dot[i, 0, points[:, 1], points[:, 0]] = 1
                        if i % 2 == 0:
                            share_mask = label[i].get("share_mask0", None)
                            if share_mask is not None and share_mask.numel() > 0:
                                share_coords = points[share_mask].long()
                                if share_coords.numel() > 0:
                                    gt_share_dot[i, 0, share_coords[:, 1].clamp(0, h - 1), share_coords[:, 0].clamp(0, w - 1)] = 1
                            out_coords = points[label[i]["outflow_mask"]].long()
                            if out_coords.numel() > 0:
                                gt_in_out_dot[i, 0, out_coords[:, 1], out_coords[:, 0]] = 1
                        else:
                            share_mask = label[i].get("share_mask1", None)
                            if share_mask is not None and share_mask.numel() > 0:
                                share_coords = points[share_mask].long()
                                if share_coords.numel() > 0:
                                    gt_share_dot[i, 0, share_coords[:, 1].clamp(0, h - 1), share_coords[:, 0].clamp(0, w - 1)] = 1
                            in_coords = points[label[i]["inflow_mask"]].long()
                            if in_coords.numel() > 0:
                                gt_in_out_dot[i, 0, in_coords[:, 1], in_coords[:, 0]] = 1

                    gt_global_den = model.gaussian_smoother(gt_global_dot)
                    _ = model.gaussian_smoother(gt_share_dot)
                    gt_in_out_den = model.gaussian_smoother(gt_in_out_dot)

                with autocast("cuda", enabled=device.type == "cuda"):
                    pre_global_den, pre_share_den, pre_in_out_den = model.test_forward(img, flow)

                pre_global_den = pre_global_den.float()
                pre_share_den = pre_share_den.float()
                pre_in_out_den = pre_in_out_den.float()
                pre_in_out_den[pre_in_out_den < 0] = 0

                pred_cnt = pre_global_den[0].sum().item()
                if vi == 0:
                    pred_dict["first_frame"] = pred_cnt

                pred_dict["inflow"].append(pre_in_out_den[1].sum().item())
                pred_dict["outflow"].append(pre_in_out_den[0].sum().item())

                if label is not None:
                    gt_count = gt_global_den[0].sum().item()
                    sing_cnt_errors["mae"].update(abs(gt_count - pred_cnt))
                    sing_cnt_errors["mse"].update((gt_count - pred_cnt) ** 2)
                    if vi == 0:
                        gt_dict["first_frame"] = gt_count
                    gt_dict["inflow"].append(gt_in_out_den[1].sum().item())
                    gt_dict["outflow"].append(gt_in_out_den[0].sum().item())

                    if opt.save_visual and vi % opt.test_intervals == 0:
                        if vi == 0:
                            prev_gt_in_out_den = gt_in_out_den
                            prev_pre_in_out_den = pre_in_out_den
                            gt_in_den = deepcopy(placeholder)
                            pre_in_den = deepcopy(placeholder)
                        else:
                            gt_in_den = prev_gt_in_out_den[1]
                            pre_in_den = prev_pre_in_out_den[1]
                        gt_share_den_before = deepcopy(placeholder)
                        pre_share_den_before = deepcopy(placeholder)
                        gt_share_den_next = deepcopy(placeholder)
                        pre_share_den_next = pre_share_den[0]
                        gt_out_den = gt_in_out_den[0]
                        pre_out_den = pre_in_out_den[0]
                        visual_maps = torch.stack(
                            [
                                gt_global_den[0],
                                pre_global_den[0],
                                gt_share_den_before,
                                pre_share_den_before,
                                gt_in_den,
                                pre_in_den,
                                gt_share_den_next,
                                pre_share_den_next,
                                gt_out_den,
                                pre_out_den,
                            ],
                            dim=0,
                        )
                        save_test_visual(visual_maps.unsqueeze(0), [img[0]], scene_name, restore_transform, opt.output_dir, 0, 0)
                        prev_gt_in_out_den = gt_in_out_den
                        prev_pre_in_out_den = pre_in_out_den

        if has_frame_gt:
            scenes_pred_dict["all"].append(pred_dict)
            scenes_gt_dict["all"].append(gt_dict)
            if scene_key in density_labels:
                lvl = density_labels[scene_key][0]
                group_key = f"density{lvl}"
                if group_key in scenes_pred_dict:
                    scenes_pred_dict[group_key].append(pred_dict)
                    scenes_gt_dict[group_key].append(gt_dict)
        else:
            if scene_gt_count is None:
                raise ValueError(f"Missing hard-coded CroHD/HT21 GT count for scene: {scene_key}")
            pred_scene_count = _pred_scene_count(pred_dict, intervals)
            scene_records.append(
                {
                    "scene_name": scene_key,
                    "pred": pred_scene_count,
                    "gt": float(scene_gt_count),
                    "abs_err": abs(pred_scene_count - scene_gt_count),
                    "time": float(video_time),
                }
            )

    if has_frame_gt:
        _write_density_results(opt.output_dir, density_keys, scenes_pred_dict, scenes_gt_dict, intervals)
        mae = sing_cnt_errors["mae"].avg
        mse = np.sqrt(sing_cnt_errors["mse"].avg)
        print("frame_mae: %.2f, frame_mse: %.2f" % (mae, mse))
        return

    _write_scene_results(opt.output_dir, scene_records)

if __name__ == "__main__":
    from importlib import import_module

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    data_mode = opt.DATASET
    datasetting = import_module(f"datasets.setting.{data_mode}")
    cfg_data = datasetting.cfg_data

    test(cfg_data)
