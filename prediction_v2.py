import os
import argparse
import json
from time import perf_counter
from datetime import datetime
from model.pred_func import *
from model.config import load_config
from model.genconvit import GenConViT
from model.genconvit_v2 import GenConViTV2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    return acc, prec, rec, f1

def vids(
    ed_weight, vae_weight, root_dir="sample_prediction_data", dataset=None, 
    num_frames=15, net=None, fp16=False, arch_type='original', 
    use_attention=True, use_residual=True
):
    result = set_result()
    r = 0
    f = 0
    count = 0
    y_true = []
    y_pred = []
    
    model = load_model(config, arch_type, net, ed_weight, vae_weight, fp16, use_attention, use_residual)

    for filename in os.listdir(root_dir):
        curr_vid = os.path.join(root_dir, filename)

        try:
            if is_video(curr_vid):
                # For sample data, assume filenames containing 'fake' are FAKE, else REAL
                gt_label = 1 if 'fake' in filename.lower() else 0
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
                pred_label = 1 if real_or_fake(pred[0]) == 'FAKE' else 0
                y_true.append(gt_label)
                y_pred.append(pred_label)
                f, r = (f + 1, r) if pred_label == 1 else (f, r + 1)
                print(
                    f"Prediction: {pred[1]} {real_or_fake(pred[0])} \t\tFake: {f} Real: {r}"
                )
            else:
                print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    # Print metrics
    if y_true and y_pred:
        result = update_result_with_metrics(result, y_true, y_pred)
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
    y_true = []
    y_pred = []

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
            label = 0 if klass == "original" else 1  # 0: REAL, 1: FAKE
            for filename in filenames:
                try:
                    if filename in ff_file:
                        curr_vid = os.path.join(dirpath, filename)
                        compression = "c23" if "c23" in curr_vid else "c40"
                        if is_video(curr_vid):
                            result, accuracy, count, pred = predict(
                                curr_vid,
                                model,
                                fp16,
                                result,
                                num_frames,
                                net,
                                klass,
                                count,
                                accuracy,
                                "REAL" if label == 0 else "FAKE",
                                compression,
                            )
                            pred_label = 1 if real_or_fake(pred[0]) == 'FAKE' else 0
                            y_true.append(label)
                            y_pred.append(pred_label)
                        else:
                            print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")

                except Exception as e:
                    print(f"An error occurred: {str(e)}")
    if y_true and y_pred:
        result = update_result_with_metrics(result, y_true, y_pred)
    return result


def dfdc(
    ed_weight, vae_weight, root_dir="DFDC\\dfdc_test", dataset=None, 
    num_frames=15, net=None, fp16=False, arch_type='original', 
    use_attention=True, use_residual=True
):
    result = set_result()
    count = 0
    accuracy = 0
    y_true = []
    y_pred = []

    model = load_model(config, arch_type, net, ed_weight, vae_weight, fp16, use_attention, use_residual)

    # load files used in the test set from DFDC
    with open(os.path.join("json_file", "dfdc_files.json")) as j_file:
        dfdc_file = list(json.load(j_file))

    for filename in os.listdir(root_dir):
        if filename.endswith(".mp4") and filename in dfdc_file:
            curr_vid = os.path.join(root_dir, filename)
            try:
                if is_video(curr_vid):
                    gt_label = 0 if filename.endswith("_0.mp4") else 1
                    result, accuracy, count, pred = predict(
                        curr_vid,
                        model,
                        fp16,
                        result,
                        num_frames,
                        net,
                        "dfdc",
                        count,
                        accuracy,
                        "REAL" if gt_label == 0 else "FAKE",
                    )
                    pred_label = 1 if real_or_fake(pred[0]) == 'FAKE' else 0
                    y_true.append(gt_label)
                    y_pred.append(pred_label)
                else:
                    print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")
            except Exception as e:
                print(f"An error occurred: {str(e)}")
    if y_true and y_pred:
        result = update_result_with_metrics(result, y_true, y_pred)
    return result


def timit(
    ed_weight, vae_weight, root_dir="DeepfakeTIMIT", dataset=None, 
    num_frames=15, net=None, fp16=False, arch_type='original', 
    use_attention=True, use_residual=True
):
    result = set_result()
    count = 0
    accuracy = 0
    y_true = []
    y_pred = []

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
        label = 0 if klass == "real" else 1
        for filename in filenames:
            if filename.endswith(".mp4"):
                curr_vid = os.path.join(dirpath, filename)
                try:
                    if is_video(curr_vid):
                        result, accuracy, count, pred = predict(
                            curr_vid,
                            model,
                            fp16,
                            result,
                            num_frames,
                            net,
                            klass,
                            count,
                            accuracy,
                            "REAL" if label == 0 else "FAKE",
                        )
                        pred_label = 1 if real_or_fake(pred[0]) == 'FAKE' else 0
                        y_true.append(label)
                        y_pred.append(pred_label)
                    else:
                        print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")
                except Exception as e:
                    print(f"An error occurred: {str(e)}")
    if y_true and y_pred:
        acc, prec, rec, f1 = compute_metrics(y_true, y_pred)
        print(f"\nOverall Results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-score: {f1:.4f}")
    return result


def celeb(
    ed_weight, vae_weight, root_dir="Celeb-DF", dataset=None, 
    num_frames=15, net=None, fp16=False, arch_type='original', 
    use_attention=True, use_residual=True
):
    result = set_result()
    count = 0
    accuracy = 0
    y_true = []
    y_pred = []

    model = load_model(config, arch_type, net, ed_weight, vae_weight, fp16, use_attention, use_residual)

    # load files used in the test set from celeb
    with open(os.path.join("json_file", "celeb_test.json")) as j_file:
        cfl = list(json.load(j_file))

    for ck in cfl:
        ck_ = ck.split("/")
        klass = ck_[0]
        filename = ck_[1]
        correct_label = 1 if klass == "Celeb-synthesis" else 0
        vid = os.path.join(root_dir, ck)
        try:
            if is_video(vid):
                result, accuracy, count, pred = predict(
                    vid,
                    model,
                    fp16,
                    result,
                    num_frames,
                    net,
                    klass,
                    count,
                    accuracy,
                    "FAKE" if correct_label == 1 else "REAL",
                )
                pred_label = 1 if real_or_fake(pred[0]) == 'FAKE' else 0
                y_true.append(correct_label)
                y_pred.append(pred_label)
            else:
                print(f"Invalid video file: {vid}. Please provide a valid video file.")
        except Exception as e:
            print(f"An error occurred x: {str(e)}")
    if y_true and y_pred:
        acc, prec, rec, f1 = compute_metrics(y_true, y_pred)
        print(f"\nOverall Results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-score: {f1:.4f}")
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
        "--arch-type", type=str, choices=['original', 'v2'], default='original',
        help="Architecture type: original or v2 (modified activations only)"
    )
    parser.add_argument("--use-attention", action="store_true", help="Enable attention mechanism (ignored in current V2 implementation)")
    parser.add_argument("--use-residual", action="store_true", help="Enable residual connections (ignored in current V2 implementation)")

    args = parser.parse_args()
    path = args.p
    num_frames = args.f if args.f else 15
    dataset = args.d if args.d else "other"
    fp16 = True if args.fp16 else False
    arch_type = args.arch_type
    use_attention = args.use_attention
    use_residual = args.use_residual

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


def update_result_with_metrics(result, y_true, y_pred):
    """
    Update result dictionary with performance metrics
    
    Args:
        result: The result dictionary to update
        y_true: List of ground truth labels (0 for real, 1 for fake)
        y_pred: List of predicted labels (0 for real, 1 for fake)
        
    Returns:
        Updated result dictionary with metrics added
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return result
        
    acc, prec, rec, f1 = compute_metrics(y_true, y_pred)
    print(f"\nOverall Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # Add metrics to result dictionary
    result["metrics"] = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "total_samples": len(y_true),
        "true_fake": int(sum(y_true)),
        "true_real": int(len(y_true) - sum(y_true)),
        "predicted_fake": int(sum(y_pred)),
        "predicted_real": int(len(y_pred) - sum(y_pred))
    }
    
    return result


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

    # Add metadata about the run
    if "metadata" not in result:
        result["metadata"] = {}
    
    result["metadata"].update({
        "architecture": arch_type,
        "network": net,
        "dataset": dataset,
        "frames_processed": num_frames,
        "ed_weight": ed_weight,
        "vae_weight": vae_weight,
        "fp16": fp16,
        "timestamp": datetime.now().isoformat(),
        "use_attention": use_attention if arch_type == "v2" else None,
        "use_residual": use_residual if arch_type == "v2" else None
    })

    curr_time = datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
    file_path = os.path.join("result", f"prediction_{dataset}_{net}_{arch_type}_{curr_time}.json")

    with open(file_path, "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"\nResults saved to {file_path}")
    
    end_time = perf_counter()
    run_time = end_time - start_time
    print("\n\n--- %s seconds ---" % (run_time))
    
    # Add runtime to the result file
    result["metadata"]["runtime_seconds"] = run_time
    with open(file_path, "w") as f:
        json.dump(result, f, indent=2)
        
    return file_path


if __name__ == "__main__":
    main()
