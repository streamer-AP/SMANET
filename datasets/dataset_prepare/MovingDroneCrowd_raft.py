import argparse
import os
import re

import cv2
import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from tqdm import tqdm

def get_image_paths(directory):
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(".jpg")]
    image_files.sort(key=lambda f: int(re.search(r"(\d+)\.jpg$", f, re.IGNORECASE).group(1)))
    return [os.path.join(directory, f) for f in image_files]

def preprocess_image_cv2(img_path, target_size):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

def read_allowed_scenes(train_txt):
    if not os.path.exists(train_txt):
        print(f"Warning: split file not found: {train_txt}; processing all scenes.")
        return None
    with open(train_txt, "r", encoding="utf-8") as f:
        scenes = {line.strip() for line in f if line.strip()}
    print(f"Loaded {len(scenes)} scenes from {train_txt}.")
    return scenes

def collect_image_folders(input_root, allowed_scenes):
    image_folders = []
    for scene_dir in sorted(os.listdir(input_root)):
        scene_path = os.path.join(input_root, scene_dir)
        if not os.path.isdir(scene_path) or not scene_dir.startswith("scene_"):
            continue
        if allowed_scenes is not None and scene_dir not in allowed_scenes:
            continue
        for clip_dir in sorted(os.listdir(scene_path)):
            clip_path = os.path.join(scene_path, clip_dir)
            if os.path.isdir(clip_path):
                image_folders.append(clip_path)
    return image_folders

def build_pairs(image_paths, min_diff, max_diff):
    pairs = []
    frame_ids = [int(os.path.splitext(os.path.basename(path))[0]) for path in image_paths]
    for i, path_a in enumerate(image_paths):
        for j in range(i + 1, len(image_paths)):
            diff = frame_ids[j] - frame_ids[i]
            if min_diff <= diff <= max_diff:
                pairs.append((path_a, image_paths[j], frame_ids[i], frame_ids[j]))
    return pairs

def print_stats(image_folders, input_root, min_diff, max_diff):
    total_images = 0
    total_pairs = 0
    print("\n" + "=" * 50)
    print("Precompute summary")
    print(f"Frame interval range: [{min_diff}, {max_diff}]")
    print(f"Video clips: {len(image_folders)}")

    for folder_path in tqdm(image_folders, desc="Scanning clips"):
        image_paths = get_image_paths(folder_path)
        total_images += len(image_paths)
        pair_count = len(build_pairs(image_paths, min_diff, max_diff))
        total_pairs += pair_count
        print(f"{os.path.relpath(folder_path, input_root)}: {len(image_paths)} frames, {pair_count} pairs")

    print(f"Total images: {total_images}")
    print(f"Forward flow files: {total_pairs}")
    print(f"Backward flow files: {total_pairs}")
    print(f"Total flow files: {total_pairs * 2}")
    print("=" * 50 + "\n")

def load_raft_model(args, device):
    if args.raft_weights:
        model = raft_large(weights=None, progress=False).to(device)
        model.load_state_dict(torch.load(args.raft_weights, map_location="cpu"))
        transforms = lambda img1, img2: (img1, img2)
    else:
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=True).to(device)
        transforms = weights.transforms()
    model.eval()
    return model, transforms

def save_flow_batch(model, transforms, pairs, output_folder, target_size, flow_size, device):
    pending = []
    for pair in pairs:
        _, _, frame_a, frame_b = pair
        forward_flow_file = os.path.join(output_folder, f"{frame_a}_to_{frame_b}.pt")
        backward_flow_file = os.path.join(output_folder, f"{frame_b}_to_{frame_a}.pt")
        if not (os.path.exists(forward_flow_file) and os.path.exists(backward_flow_file)):
            pending.append((pair, forward_flow_file, backward_flow_file))

    if not pending:
        return 0

    img1 = torch.stack([preprocess_image_cv2(pair[0], target_size) for pair, _, _ in pending]).to(device)
    img2 = torch.stack([preprocess_image_cv2(pair[1], target_size) for pair, _, _ in pending]).to(device)
    img1, img2 = transforms(img1, img2)
    img1 = img1.contiguous()
    img2 = img2.contiguous()

    with torch.no_grad():
        forward_flow = model(img1, img2)[-1]
        backward_flow = model(img2, img1)[-1]

    forward_flow = torch.nn.functional.interpolate(
        forward_flow, size=flow_size, mode="bilinear", align_corners=False
    ).cpu()
    backward_flow = torch.nn.functional.interpolate(
        backward_flow, size=flow_size, mode="bilinear", align_corners=False
    ).cpu()

    for idx, (_, forward_flow_file, backward_flow_file) in enumerate(pending):
        torch.save(forward_flow[idx : idx + 1], forward_flow_file)
        torch.save(backward_flow[idx : idx + 1], backward_flow_file)
    return len(pending)

def parse_args():
    parser = argparse.ArgumentParser(description="Precompute RAFT optical flow for MovingDroneCrowd.")
    parser.add_argument("--data_root", default=os.environ.get("MDC_DATA_ROOT", "./data/MovingDroneCrowd"))
    parser.add_argument("--output_root", default=os.environ.get("MDC_FLOW_ROOT"))
    parser.add_argument("--train_txt", default=None)
    parser.add_argument("--raft_weights", default=os.environ.get("RAFT_WEIGHTS"))
    parser.add_argument("--min_frame_diff", type=int, default=3)
    parser.add_argument("--max_frame_diff", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--target_h", type=int, default=768)
    parser.add_argument("--target_w", type=int, default=1024)
    parser.add_argument("--flow_h", type=int, default=96)
    parser.add_argument("--flow_w", type=int, default=128)
    parser.add_argument("--stats_only", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    input_root = os.path.join(args.data_root, "frames")
    output_root = args.output_root or os.path.join(args.data_root, "raft")
    train_txt = args.train_txt or os.path.join(args.data_root, "train.txt")
    target_size = (args.target_h, args.target_w)
    flow_size = (args.flow_h, args.flow_w)

    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Frame root not found: {input_root}")

    allowed_scenes = read_allowed_scenes(train_txt)
    image_folders = collect_image_folders(input_root, allowed_scenes)
    print_stats(image_folders, input_root, args.min_frame_diff, args.max_frame_diff)
    if args.stats_only:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading raft_large model...")
    model, transforms = load_raft_model(args, device)

    with tqdm(total=len(image_folders), desc="Clips") as pbar_total:
        for folder_path in image_folders:
            relative_path = os.path.relpath(folder_path, input_root)
            output_folder = os.path.join(output_root, relative_path)
            os.makedirs(output_folder, exist_ok=True)
            image_paths = get_image_paths(folder_path)
            pairs = build_pairs(image_paths, args.min_frame_diff, args.max_frame_diff)

            with tqdm(total=len(pairs), desc=f"Flow {relative_path}", leave=False) as pbar_pairs:
                for start in range(0, len(pairs), args.batch_size):
                    batch = pairs[start : start + args.batch_size]
                    save_flow_batch(model, transforms, batch, output_folder, target_size, flow_size, device)
                    pbar_pairs.update(len(batch))
            pbar_total.update(1)

    print("Optical-flow precomputation complete.")

if __name__ == "__main__":
    main()
