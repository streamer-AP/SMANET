import argparse
import os
from pathlib import Path

import cv2
import torch
import torch.multiprocessing as mp
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from tqdm import tqdm

cv2.setNumThreads(1)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute RAFT optical flow for SenseCrowd.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_root", default=os.environ.get("SENSE_VIDEO_ROOT", "./data/SENSE/videos"))
    parser.add_argument("--output_root", default=os.environ.get("SENSE_FLOW_ROOT", "./data/SENSE/raft"))
    parser.add_argument("--split_txt", default=os.environ.get("SENSE_SPLIT_TXT", "./data/SENSE/test.txt"))
    parser.add_argument("--min_diff", type=int, default=10)
    parser.add_argument("--max_diff", type=int, default=20)
    parser.add_argument("--target_h", type=int, default=768)
    parser.add_argument("--target_w", type=int, default=1024)
    parser.add_argument("--save_h", type=int, default=96)
    parser.add_argument("--save_w", type=int, default=128)
    parser.add_argument("--raft_weights", default=os.environ.get("RAFT_WEIGHTS", ""))
    parser.add_argument(
        "--allow_non_strict_weights",
        action="store_true",
        help="Allow missing/unexpected keys when loading a custom RAFT checkpoint.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    parser.add_argument("--stats_only", action="store_true")
    return parser.parse_args()

def read_split_video_ids(split_txt):
    if not split_txt or not os.path.exists(split_txt):
        return None

    video_ids = []
    seen = set()
    with open(split_txt, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            video_id = fields[0]
            if video_id in seen:
                print(f"Warning: duplicate split entry at line {line_no}: {video_id}")
                continue
            seen.add(video_id)
            video_ids.append(video_id)
    return video_ids

def load_video_frames(video_folder, video_id):
    frames = []
    for name in os.listdir(video_folder):
        if not name.lower().endswith(".jpg"):
            continue
        try:
            frame_id = int(Path(name).stem)
        except ValueError:
            print(f"Warning: skip non-numeric frame name in {video_id}: {name}")
            continue
        frames.append((frame_id, os.path.join(video_folder, name)))

    frames.sort(key=lambda item: item[0])
    deduped = []
    seen = set()
    for frame_id, path in frames:
        if frame_id in seen:
            print(f"Warning: duplicate frame id in {video_id}: {frame_id}")
            continue
        seen.add(frame_id)
        deduped.append((frame_id, path))
    return deduped

def collect_videos(input_root, split_txt):
    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Video root not found: {input_root}")

    split_ids = read_split_video_ids(split_txt)
    if split_ids is None:
        print(f"Warning: split file not found: {split_txt}; processing all videos.")
        split_ids = sorted(name for name in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, name)))

    videos = []
    stats = {
        "split_videos": len(split_ids),
        "missing_video_folders": 0,
        "videos_with_no_valid_frames": 0,
        "videos_with_insufficient_frames": 0,
        "usable_videos": 0,
        "total_frames": 0,
    }

    for video_id in split_ids:
        video_folder = os.path.join(input_root, video_id)
        if not os.path.isdir(video_folder):
            print(f"Warning: missing video folder: {video_folder}")
            stats["missing_video_folders"] += 1
            continue

        frames = load_video_frames(video_folder, video_id)
        if not frames:
            stats["videos_with_no_valid_frames"] += 1
            continue
        if len(frames) < 2:
            stats["videos_with_insufficient_frames"] += 1
            continue

        videos.append((video_id, frames))
        stats["usable_videos"] += 1
        stats["total_frames"] += len(frames)
    return videos, stats

def build_pairs(video_id, frames, output_root, min_diff, max_diff, skip_existing):
    out_dir = os.path.join(output_root, video_id)
    os.makedirs(out_dir, exist_ok=True)

    pairs = []
    candidate_count = 0
    skipped_count = 0
    for i, (frame_a, path_a) in enumerate(frames):
        for frame_b, path_b in frames[i + 1 :]:
            diff = frame_b - frame_a
            if diff < min_diff:
                continue
            if diff > max_diff:
                break

            candidate_count += 1
            forward_path = os.path.join(out_dir, f"{frame_a}_to_{frame_b}.pt")
            backward_path = os.path.join(out_dir, f"{frame_b}_to_{frame_a}.pt")
            if skip_existing and os.path.exists(forward_path) and os.path.exists(backward_path):
                skipped_count += 1
                continue
            pairs.append((path_a, path_b, frame_a, frame_b, video_id))
    return pairs, candidate_count, skipped_count

def preprocess_image(path, target_h, target_w):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)

def load_raft_model(args, device):
    if args.raft_weights:
        if not os.path.exists(args.raft_weights):
            raise FileNotFoundError(f"RAFT checkpoint not found: {args.raft_weights}")
        model = raft_large(weights=None, progress=False).to(device)
        state_dict = torch.load(args.raft_weights, map_location=device)
        load_result = model.load_state_dict(state_dict, strict=not args.allow_non_strict_weights)
        if args.allow_non_strict_weights and (load_result.missing_keys or load_result.unexpected_keys):
            print(f"Warning: RAFT missing keys: {load_result.missing_keys}")
            print(f"Warning: RAFT unexpected keys: {load_result.unexpected_keys}")
        transforms = lambda img_a, img_b: (img_a, img_b)
    else:
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=False).to(device)
        transforms = weights.transforms()

    if args.fp16:
        model = model.half()
    model.eval()
    return model, transforms

def run_raft_on_batch(model, transforms, batch_pairs, device, args):
    imgs_a = [preprocess_image(pair[0], args.target_h, args.target_w) for pair in batch_pairs]
    imgs_b = [preprocess_image(pair[1], args.target_h, args.target_w) for pair in batch_pairs]
    batch_a = torch.stack(imgs_a, dim=0).to(device, non_blocking=True).contiguous()
    batch_b = torch.stack(imgs_b, dim=0).to(device, non_blocking=True).contiguous()
    batch_a, batch_b = transforms(batch_a, batch_b)

    if args.fp16:
        batch_a = batch_a.half()
        batch_b = batch_b.half()

    with torch.no_grad():
        src = torch.cat([batch_a, batch_b], dim=0)
        dst = torch.cat([batch_b, batch_a], dim=0)
        flows = model(src, dst)[-1]
        pair_count = len(batch_pairs)
        forward = flows[:pair_count]
        backward = flows[pair_count:]

    forward = torch.nn.functional.interpolate(
        forward.float(), size=(args.save_h, args.save_w), mode="bilinear", align_corners=False
    )
    backward = torch.nn.functional.interpolate(
        backward.float(), size=(args.save_h, args.save_w), mode="bilinear", align_corners=False
    )
    return forward, backward

def save_pair_flow(forward, backward, pair, output_root):
    _, _, frame_a, frame_b, video_id = pair
    out_dir = os.path.join(output_root, video_id)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(forward.unsqueeze(0).cpu(), os.path.join(out_dir, f"{frame_a}_to_{frame_b}.pt"))
    torch.save(backward.unsqueeze(0).cpu(), os.path.join(out_dir, f"{frame_b}_to_{frame_a}.pt"))

def print_stats(videos, stats, args):
    total_candidates = 0
    total_skipped = 0
    for video_id, frames in videos:
        _, candidates, skipped = build_pairs(
            video_id, frames, args.output_root, args.min_diff, args.max_diff, args.skip_existing
        )
        total_candidates += candidates
        total_skipped += skipped

    print("=" * 80)
    print("SenseCrowd RAFT precompute summary")
    print(f"input_root          : {args.input_root}")
    print(f"output_root         : {args.output_root}")
    print(f"split_txt           : {args.split_txt}")
    print(f"min_diff/max_diff   : {args.min_diff}/{args.max_diff}")
    print(f"target_h/target_w   : {args.target_h}/{args.target_w}")
    print(f"save_h/save_w       : {args.save_h}/{args.save_w}")
    print(f"batch_size          : {args.batch_size}")
    print(f"num_gpus            : {args.num_gpus}")
    print(f"gpu_ids             : {args.gpu_ids}")
    print(f"fp16                : {args.fp16}")
    print(f"skip_existing       : {args.skip_existing}")
    print("-" * 80)
    print(f"split videos        : {stats['split_videos']}")
    print(f"usable videos       : {stats['usable_videos']}")
    print(f"missing folders     : {stats['missing_video_folders']}")
    print(f"no valid frames     : {stats['videos_with_no_valid_frames']}")
    print(f"insufficient frames : {stats['videos_with_insufficient_frames']}")
    print(f"total frames        : {stats['total_frames']}")
    print(f"candidate pairs     : {total_candidates}")
    print(f"skip-existing pairs : {total_skipped}")
    print(f"pairs to process    : {total_candidates - total_skipped}")
    print(f"expected .pt files  : {(total_candidates - total_skipped) * 2}")
    print("=" * 80)

def select_gpu_ids(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for optical-flow generation.")

    available = torch.cuda.device_count()
    candidates = args.gpu_ids if args.gpu_ids is not None else list(range(available))
    candidates = [idx for idx in candidates if 0 <= idx < available]
    candidates = candidates[: max(1, args.num_gpus)]
    if not candidates:
        raise RuntimeError("No usable GPU ids were selected.")
    return candidates

def worker(rank, videos, args, gpu_ids):
    gpu_id = gpu_ids[rank]
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)
    torch.backends.cudnn.benchmark = True

    model, transforms = load_raft_model(args, device)
    assigned_videos = videos[rank::len(gpu_ids)]
    progress = tqdm(total=len(assigned_videos), desc=f"GPU{gpu_id}", position=rank)

    saved_pairs = 0
    failed_pairs = 0
    for video_id, frames in assigned_videos:
        pairs, _, _ = build_pairs(video_id, frames, args.output_root, args.min_diff, args.max_diff, args.skip_existing)
        for start in range(0, len(pairs), args.batch_size):
            batch_pairs = pairs[start : start + args.batch_size]
            try:
                forward, backward = run_raft_on_batch(model, transforms, batch_pairs, device, args)
                for idx, pair in enumerate(batch_pairs):
                    save_pair_flow(forward[idx], backward[idx], pair, args.output_root)
                saved_pairs += len(batch_pairs)
            except Exception as exc:
                print(f"Warning: batch failed in {video_id}; retrying single pairs. Error: {exc}")
                for pair in batch_pairs:
                    try:
                        forward, backward = run_raft_on_batch(model, transforms, [pair], device, args)
                        save_pair_flow(forward[0], backward[0], pair, args.output_root)
                        saved_pairs += 1
                    except Exception as single_exc:
                        failed_pairs += 1
                        print(f"Warning: failed pair {pair[4]} {pair[2]}_to_{pair[3]}: {single_exc}")
        progress.update(1)

    progress.close()
    print(f"GPU {gpu_id} done: saved_pairs={saved_pairs}, failed_pairs={failed_pairs}")

def validate_args(args):
    if args.min_diff <= 0 or args.max_diff <= 0:
        raise ValueError("Frame differences must be positive.")
    if args.min_diff > args.max_diff:
        raise ValueError("min_diff cannot be larger than max_diff.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

def main():
    args = parse_args()
    validate_args(args)
    os.makedirs(args.output_root, exist_ok=True)

    videos, stats = collect_videos(args.input_root, args.split_txt)
    if not videos:
        print("No usable videos found.")
        return

    print_stats(videos, stats, args)
    if args.stats_only:
        return

    gpu_ids = select_gpu_ids(args)
    if not args.raft_weights:
        _ = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True)
    print(f"Using {len(gpu_ids)} GPU process(es): {gpu_ids}")
    mp.spawn(worker, nprocs=len(gpu_ids), args=(videos, args, gpu_ids), join=True)
    print("Optical-flow precomputation complete.")

if __name__ == "__main__":
    main()
