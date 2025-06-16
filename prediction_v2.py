import os
import argparse
import json
from time import perf_counter
from datetime import datetime
from model.pred_func import *
from model.config import load_config
from model.genconvit import GenConViT
from model.genconvit_v2 import GenConViTV2

config = load_config()
print('CONFIG')
print(config)

def load_model(config, arch_type, net, ed_weight, vae_weight, fp16, use_attention=True, use_residual=True):
    """
    Load either the original GenConViT model or the enhanced V2 architecture
    
    Args:
        config: Model configuration dictionary
        arch_type: Architecture type ('original' or 'v2')
        net: Network type ('ed', 'vae', 'genconvit', or 'v2')
        ed_weight: Path to ED model weights
        vae_weight: Path to VAE model weights
        fp16: Whether to use half precision
        use_attention: Whether to use attention mechanism (for V2 only)
        use_residual: Whether to use residual connections (for V2 only)
    
    Returns:
        Loaded model
    """
    if arch_type == 'v2':
        print(f"Loading GenConViTV2 with net={net}, attention={use_attention}, residual={use_residual}")
        return GenConViTV2(config, ed_weight, vae_weight, net, fp16, use_attention, use_residual)
    else:
        print(f"Loading original GenConViT with net={net}")
        return GenConViT(config, ed_weight, vae_weight, net, fp16)

def vids(
    ed_weight, vae_weight, root_dir="sample_prediction_data", dataset=None, 
    num_frames=15, net=None, fp16=False, arch_type='original', 
    use_attention=True, use_residual=True
):
    result = set_result()
    r = 0
    f = 0
    count = 0
    
    model = load_model(config, arch_type, net, ed_weight, vae_weight, fp16, use_attention, use_residual)

    for filename in os.listdir(root_dir):
        curr_vid = os.path.join(root_dir, filename)

        try:
            if is_video(curr_vid):
                result, accuracy, count, pred = predict(
                    curr_vid,
                    model,
                    fp16,
                    result,
                    num_frames,
                    net,
                    "uncategorized",
                    count,
                )
                f, r = (f + 1, r) if "FAKE" == real_or_fake(pred[0]) else (f, r + 1)
                print(
                    f"Prediction: {pred[1]} {real_or_fake(pred[0])} \t\tFake: {f} Real: {r}"
                )
            else:
                print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    return result


def faceforensics(
    ed_weight, vae_weight, root_dir="FaceForensics\\data", dataset=None, 
    num_frames=15, net=None, fp16=False, arch_type='original', 
    use_attention=True, use_residual=True
):
    vid_type = ["original_sequences", "manipulated_sequences"]
    result = set_result()
    result["video"]["compression"] = []
    ffdirs = [
        "DeepFakeDetection",
        "Deepfakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
    ]

    # load files not used in the training set, the files are appended with compression type, _c23 or _c40
    with open(os.path.join("json_file", "ff_file_list.json")) as j_file:
        ff_file = list(json.load(j_file))

    count = 0
    accuracy = 0
    model = load_model(config, arch_type, net, ed_weight, vae_weight, fp16, use_attention, use_residual)

    for v_t in vid_type:
        for dirpath, dirnames, filenames in os.walk(os.path.join(root_dir, v_t)):
            klass = next(
                filter(lambda x: x in dirpath.split(os.path.sep), ffdirs),
                "original",
            )
            label = "REAL" if klass == "original" else "FAKE"
            for filename in filenames:
                try:
                    if filename in ff_file:
                        curr_vid = os.path.join(dirpath, filename)
                        compression = "c23" if "c23" in curr_vid else "c40"
                        if is_video(curr_vid):
                            result, accuracy, count, _ = predict(
                                curr_vid,
                                model,
                                fp16,
                                result,
                                num_frames,
                                net,
                                klass,
                                count,
                                accuracy,
                                label,
                                compression,
                            )
                        else:
                            print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")

                except Exception as e:
                    print(f"An error occurred: {str(e)}")

    return result


def dfdc(
    ed_weight, vae_weight, root_dir="DFDC\\dfdc_test", dataset=None, 
    num_frames=15, net=None, fp16=False, arch_type='original', 
    use_attention=True, use_residual=True
):
    result = set_result()
    count = 0
    accuracy = 0

    model = load_model(config, arch_type, net, ed_weight, vae_weight, fp16, use_attention, use_residual)

    # load files used in the test set from DFDC
    with open(os.path.join("json_file", "dfdc_files.json")) as j_file:
        dfdc_file = list(json.load(j_file))

    for filename in os.listdir(root_dir):
        if filename.endswith(".mp4") and filename in dfdc_file:
            curr_vid = os.path.join(root_dir, filename)

            try:
                if is_video(curr_vid):
                    result, accuracy, count, _ = predict(
                        curr_vid,
                        model,
                        fp16,
                        result,
                        num_frames,
                        net,
                        "dfdc",
                        count,
                        accuracy,
                        "REAL" if filename.endswith("_0.mp4") else "FAKE",
                    )
                else:
                    print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")

            except Exception as e:
                print(f"An error occurred: {str(e)}")

    return result


def timit(
    ed_weight, vae_weight, root_dir="DeepfakeTIMIT", dataset=None, 
    num_frames=15, net=None, fp16=False, arch_type='original', 
    use_attention=True, use_residual=True
):
    result = set_result()
    count = 0
    accuracy = 0

    model = load_model(config, arch_type, net, ed_weight, vae_weight, fp16, use_attention, use_residual)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # lower quality -> lq, higher quality -> hq
        klass = (
            "low_quality"
            if "lower_quality" in dirpath.split(os.path.sep)
            else "high_quality"
            if "higher_quality" in dirpath.split(os.path.sep)
            else "real"
        )
        label = "REAL" if klass == "real" else "FAKE"
        for filename in filenames:
            if filename.endswith(".mp4"):
                curr_vid = os.path.join(dirpath, filename)

                try:
                    if is_video(curr_vid):
                        result, accuracy, count, _ = predict(
                            curr_vid,
                            model,
                            fp16,
                            result,
                            num_frames,
                            net,
                            klass,
                            count,
                            accuracy,
                            label,
                        )
                    else:
                        print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")

                except Exception as e:
                    print(f"An error occurred: {str(e)}")

    return result


def celeb(
    ed_weight, vae_weight, root_dir="Celeb-DF", dataset=None, 
    num_frames=15, net=None, fp16=False, arch_type='original', 
    use_attention=True, use_residual=True
):
    result = set_result()
    count = 0
    accuracy = 0

    model = load_model(config, arch_type, net, ed_weight, vae_weight, fp16, use_attention, use_residual)

    # load files used in the test set from celeb
    with open(os.path.join("json_file", "celeb_test.json")) as j_file:
        cfl = list(json.load(j_file))

    for ck in cfl:
        ck_ = ck.split("/")
        klass = ck_[0]
        filename = ck_[1]
        correct_label = "FAKE" if klass == "Celeb-synthesis" else "REAL"
        vid = os.path.join(root_dir, ck)

        try:
            if is_video(vid):
                result, accuracy, count, _ = predict(
                    vid,
                    model,
                    fp16,
                    result,
                    num_frames,
                    net,
                    klass,
                    count,
                    accuracy,
                    correct_label,
                )
            else:
                print(f"Invalid video file: {vid}. Please provide a valid video file.")

        except Exception as e:
            print(f"An error occurred x: {str(e)}")

    return result


def predict(
    vid,
    model,
    fp16,
    result,
    num_frames,
    net,
    klass,
    count=0,
    accuracy=-1,
    correct_label="unknown",
    compression=None,
):
    count += 1
    print(f"\n\n{str(count)} Loading... {vid}")

    df = df_face(vid, num_frames, net)  # extract face from the frames
    if fp16:
        df.half()
    y, y_val = (
        pred_vid(df, model)
        if len(df) >= 1
        else (torch.tensor(0).item(), torch.tensor(0.5).item())
    )
    result = store_result(
        result, os.path.basename(vid), y, y_val, klass, correct_label, compression
    )

    if accuracy > -1:
        if correct_label == real_or_fake(y):
            accuracy += 1
        print(
            f"\nPrediction: {y_val} {real_or_fake(y)} \t\t {accuracy}/{count} {accuracy/count}"
        )

    return result, accuracy, count, [y, y_val]


def gen_parser():
    parser = argparse.ArgumentParser("GenConViT prediction")
    parser.add_argument("--p", type=str, help="video or image path")
    parser.add_argument(
        "--f", type=int, help="number of frames to process for prediction"
    )
    parser.add_argument(
        "--d", type=str, help="dataset type, dfdc, faceforensics, timit, celeb"
    )
    parser.add_argument(
        "--s", help="model size type: tiny, large.",
    )
    parser.add_argument(
        "--e", nargs='?', const='genconvit_ed_inference', default='genconvit_ed_inference', help="weight for ed.",
    )
    parser.add_argument(
        "--v", '--value', nargs='?', const='genconvit_vae_inference', default='genconvit_vae_inference', help="weight for vae.",
    )
    
    parser.add_argument("--fp16", type=str, help="half precision support")
    
    # New arguments for architecture selection and configuration
    parser.add_argument(
        "--arch", type=str, choices=['original', 'v2'], default='original',
        help="Architecture type: original or v2 (enhanced)"
    )
    parser.add_argument("--no-attention", action="store_true", help="Disable attention mechanism for V2 architecture")
    parser.add_argument("--no-residual", action="store_true", help="Disable residual connections for V2 architecture")

    args = parser.parse_args()
    path = args.p
    num_frames = args.f if args.f else 15
    dataset = args.d if args.d else "other"
    fp16 = True if args.fp16 else False
    arch_type = args.arch
    use_attention = not args.no_attention
    use_residual = not args.no_residual

    net = 'genconvit'
    ed_weight = 'genconvit_ed_inference'
    vae_weight = 'genconvit_vae_inference'

    if args.e and args.v:
        ed_weight = args.e
        vae_weight = args.v
    elif args.e:
        net = 'ed'
        ed_weight = args.e
    elif args.v:
        net = 'vae'
        vae_weight = args.v
    
    # Set net to 'v2' if using the v2 architecture and net is still 'genconvit'
    if arch_type == 'v2' and net == 'genconvit':
        net = 'v2'
        
    print(f'\nUsing {net} with architecture: {arch_type}\n')
    if arch_type == 'v2':
        print(f'Attention: {use_attention}, Residual: {use_residual}')

    if args.s:
        if args.s in ['tiny', 'large']:
            config["model"]["backbone"] = f"convnext_{args.s}"
            config["model"]["embedder"] = f"swin_{args.s}_patch4_window7_224"
            config["model"]["type"] = args.s
    
    return path, dataset, num_frames, net, fp16, ed_weight, vae_weight, arch_type, use_attention, use_residual


def main():
    start_time = perf_counter()
    path, dataset, num_frames, net, fp16, ed_weight, vae_weight, arch_type, use_attention, use_residual = gen_parser()
    
    # Call the appropriate dataset function with the new parameters
    if dataset in ["dfdc", "faceforensics", "timit", "celeb"]:
        result = globals()[dataset](
            ed_weight, vae_weight, path, dataset, num_frames, 
            net, fp16, arch_type, use_attention, use_residual
        )
    else:
        result = vids(
            ed_weight, vae_weight, path, dataset, num_frames, 
            net, fp16, arch_type, use_attention, use_residual
        )

    curr_time = datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
    file_path = os.path.join("result", f"prediction_{dataset}_{net}_{arch_type}_{curr_time}.json")

    with open(file_path, "w") as f:
        json.dump(result, f)
    end_time = perf_counter()
    print("\n\n--- %s seconds ---" % (end_time - start_time))


if __name__ == "__main__":
    main()
