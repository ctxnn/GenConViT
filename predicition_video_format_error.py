import os
import argparse
import json
from datetime import datetime
from time import perf_counter
import torch
from PIL import Image
from torchvision import transforms
from model.pred_func import *
from model.config import load_config

# Load config
config = load_config()
print('CONFIG', config)

def df_face_from_images(frame_paths, num_frames, net):
    to_tensor = transforms.ToTensor()
    imgs = []
    for p in frame_paths[:num_frames]:
        img = Image.open(p).convert('RGB')
        imgs.append(to_tensor(img))
    return torch.stack(imgs)  # [N, C, H, W]

def faceforensics(ed_weight, vae_weight, root_dir="FaceForensics++", dataset=None, num_frames=15, net=None, fp16=False):
    ffdirs = ["DeepFakeDetection", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
    result = set_result()
    result["video"]["compression"] = []

    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)
    count, accuracy = 0, 0

    for klass in ffdirs:
        for compression in ["c23", "c40"]:
            frame_root = os.path.join(root_dir, "manipulated_sequences", klass, compression, "frames")
            if not os.path.isdir(frame_root):
                continue

            for vid_id in os.listdir(frame_root):
                vid_folder = os.path.join(frame_root, vid_id)
                if not os.path.isdir(vid_folder):
                    continue

                frame_paths = sorted([
                    os.path.join(vid_folder, f)
                    for f in os.listdir(vid_folder)
                    if f.endswith('.png')
                ])
                if not frame_paths:
                    print(f"No frames in {vid_folder}")
                    continue

                try:
                    df = df_face_from_images(frame_paths, num_frames, net)
                    if fp16:
                        df = df.half()
                    y, y_val = pred_vid(df, model)

                    label = "FAKE"
                    result = store_result(result, vid_id, y, y_val, klass, label, compression)

                    count += 1
                    if label == real_or_fake(y):
                        accuracy += 1

                    print(f"{count}. {vid_id} [{compression}] → {y_val:.4f} {real_or_fake(y)} ({accuracy}/{count})")
                except Exception as e:
                    print(f"Error on {vid_id}: {e}")

    return result

def gen_parser():
    parser = argparse.ArgumentParser("GenConViT prediction")
    parser.add_argument("--p", type=str, help="dataset root path")
    parser.add_argument("--f", type=int, help="number of frames to process", default=15)
    parser.add_argument("--d", type=str, help="dataset type", default="faceforensics")
    parser.add_argument("--s", type=str, help="model size (tiny or large)", default=None)
    parser.add_argument("--e", type=str, help="encoder weight", default="genconvit_ed_inference")
    parser.add_argument("--v", type=str, help="vae weight", default="genconvit_vae_inference")
    parser.add_argument("--fp16", action="store_true", help="use fp16 inference")
    return parser

def main():
    args = gen_parser().parse_args()
    path = args.p
    num_frames = args.f
    dataset = args.d
    fp16 = args.fp16
    net = 'genconvit'

    if args.s in ['tiny', 'large']:
        config["model"]["backbone"] = f"convnext_{args.s}"
        config["model"]["embedder"] = f"swin_{args.s}_patch4_window7_224"
        config["model"]["type"] = args.s

    result = faceforensics(args.e, args.v, path, dataset, num_frames, net, fp16)

    os.makedirs("result", exist_ok=True)
    fname = f"prediction_{dataset}_{net}_{datetime.now():%Y_%m_%d_%H_%M_%S}.json"
    with open(os.path.join("result", fname), "w") as f:
        json.dump(result, f, indent=2)

    print(f"Done — results saved to {fname}")

if __name__ == "__main__":
    main()
