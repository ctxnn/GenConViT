import os
import argparse
import json
from time import perf_counter
from datetime import datetime
from pred_func_updated import *
from model.config import load_config

config = load_config()

def vids(
    ed_weight, vae_weight, root_dir="sample_prediction_data", dataset=None, num_frames=15, net=None, fp16=False
):
    """
    Process videos in a directory for deepfake detection
    """
    result = set_result()
    r = 0
    f = 0
    count = 0
    
    print(f"Loading {net.upper()} model...")
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)
    print("Model loaded successfully!\n")

    if not os.path.exists(root_dir):
        print(f"Error: Directory {root_dir} does not exist.")
        return result

    video_files = [f for f in os.listdir(root_dir) if is_video(os.path.join(root_dir, f))]
    
    if not video_files:
        print(f"No video files found in {root_dir}")
        return result
    
    print(f"Found {len(video_files)} video files to process\n")

    for filename in video_files:
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
                    f"Prediction: {pred[1]:.4f} {real_or_fake(pred[0])} \t\tFake: {f} Real: {r}"
                )
            else:
                print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")

        except Exception as e:
            print(f"An error occurred processing {filename}: {str(e)}")

    print(f"\nSummary: {f} fake videos, {r} real videos detected")
    return result


def faceforensics(
    ed_weight, vae_weight, root_dir="FaceForensics\\data", dataset=None, num_frames=15, net=None, fp16=False
):
    """
    Process FaceForensics dataset
    """
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

    # load files not used in the training set
    try:
        with open(os.path.join("json_file", "ff_file_list.json")) as j_file:
            ff_file = list(json.load(j_file))
    except FileNotFoundError:
        print("Warning: ff_file_list.json not found, processing all files")
        ff_file = None

    count = 0
    accuracy = 0
    
    print(f"Loading {net.upper()} model for FaceForensics...")
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)
    print("Model loaded successfully!\n")

    for v_t in vid_type:
        video_type_path = os.path.join(root_dir, v_t)
        if not os.path.exists(video_type_path):
            print(f"Warning: {video_type_path} does not exist, skipping...")
            continue
            
        for dirpath, dirnames, filenames in os.walk(video_type_path):
            klass = next(
                filter(lambda x: x in dirpath.split(os.path.sep), ffdirs),
                "original",
            )
            label = "REAL" if klass == "original" else "FAKE"
            
            for filename in filenames:
                try:
                    if ff_file is None or filename in ff_file:
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
                            print(f"Invalid video file: {curr_vid}")

                except Exception as e:
                    print(f"An error occurred processing {filename}: {str(e)}")

    return result


def timit(ed_weight, vae_weight, root_dir="DeepfakeTIMIT", dataset=None, num_frames=15, net=None, fp16=False):
    """
    Process DeepfakeTIMIT dataset
    """
    keywords = ["higher_quality", "lower_quality"]
    result = set_result()
    
    print(f"Loading {net.upper()} model for TIMIT...")
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)
    print("Model loaded successfully!\n")
    
    count = 0
    accuracy = 0
    
    for keyword in keywords:
        keyword_folder_path = os.path.join(root_dir, keyword)
        if not os.path.exists(keyword_folder_path):
            print(f"Warning: {keyword_folder_path} does not exist, skipping...")
            continue
            
        for subfolder_name in os.listdir(keyword_folder_path):
            subfolder_path = os.path.join(keyword_folder_path, subfolder_name)
            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    if filename.endswith(".avi"):
                        curr_vid = os.path.join(subfolder_path, filename)
                        try:
                            if is_video(curr_vid):
                                result, accuracy, count, _ = predict(
                                    curr_vid,
                                    model,
                                    fp16,
                                    result,
                                    num_frames,
                                    net,
                                    "DeepfakeTIMIT",
                                    count,
                                    accuracy,
                                    "FAKE",
                                )
                            else:
                                print(f"Invalid video file: {curr_vid}")

                        except Exception as e:
                            print(f"An error occurred processing {filename}: {str(e)}")

    return result


def dfdc(
    ed_weight,
    vae_weight,
    root_dir="deepfake-detection-challenge\\train_sample_videos",
    dataset=None,
    num_frames=15,
    net=None,
    fp16=False,
):
    """
    Process DFDC dataset
    """
    result = set_result()
    
    # Load DFDC file list
    try:
        with open(os.path.join("json_file", "dfdc_files.json")) as data_file:
            dfdc_data = json.load(data_file)
    except FileNotFoundError:
        print("Warning: dfdc_files.json not found, processing all files in directory")
        dfdc_data = [f for f in os.listdir(root_dir) if is_video(os.path.join(root_dir, f))]

    # Load metadata
    dfdc_meta = {}
    metadata_path = os.path.join(root_dir, "metadata.json")
    if os.path.isfile(metadata_path):
        with open(metadata_path) as data_file:
            dfdc_meta = json.load(data_file)
    else:
        print("Warning: metadata.json not found")

    print(f"Loading {net.upper()} model for DFDC...")
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)
    print("Model loaded successfully!\n")
    
    count = 0
    accuracy = 0
    
    for dfdc in dfdc_data:
        dfdc_file = os.path.join(root_dir, dfdc)

        try:
            if is_video(dfdc_file):
                label = dfdc_meta.get(dfdc, {}).get("label", "UNKNOWN")
                result, accuracy, count, _ = predict(
                    dfdc_file,
                    model,
                    fp16,
                    result,
                    num_frames,
                    net,
                    "dfdc",
                    count,
                    accuracy,
                    label,
                )
            else:
                print(f"Invalid video file: {dfdc_file}")

        except Exception as e:
            print(f"An error occurred processing {dfdc}: {str(e)}")

    return result


def celeb(ed_weight, vae_weight, root_dir="Celeb-DF-v2", dataset=None, num_frames=15, net=None, fp16=False):
    """
    Process Celeb-DF dataset
    """
    try:
        with open(os.path.join("json_file", "celeb_test.json"), "r") as f:
            cfl = json.load(f)
    except FileNotFoundError:
        print("Warning: celeb_test.json not found, processing all files")
        cfl = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if is_video(os.path.join(root, file)):
                    rel_path = os.path.relpath(os.path.join(root, file), root_dir)
                    cfl.append(rel_path.replace(os.path.sep, '/'))
    
    result = set_result()
    count = 0
    accuracy = 0
    
    print(f"Loading {net.upper()} model for Celeb-DF...")
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)
    print("Model loaded successfully!\n")

    for ck in cfl:
        ck_ = ck.split("/")
        klass = ck_[0] if len(ck_) > 1 else "unknown"
        filename = ck_[-1]
        correct_label = "FAKE" if "synthesis" in klass.lower() else "REAL"
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
                print(f"Invalid video file: {vid}")

        except Exception as e:
            print(f"An error occurred processing {filename}: {str(e)}")

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
    """
    Make prediction on a single video
    """
    count += 1
    print(f"\n{str(count)} Loading... {os.path.basename(vid)}")

    try:
        df = df_face(vid, num_frames, net)  # extract face from the frames
        
        if len(df) == 0:
            print(f"No faces detected in {vid}")
            y, y_val = 0, 0.5  # Default to REAL with low confidence
        else:
            if fp16 and df.dtype != torch.float16:
                df = df.half()
            y, y_val = pred_vid(df, model)
        
        result = store_result(
            result, os.path.basename(vid), y, y_val, klass, correct_label, compression
        )

        if accuracy > -1:
            if correct_label == real_or_fake(y):
                accuracy += 1
            print(
                f"Prediction: {y_val:.4f} {real_or_fake(y)} \t\t {accuracy}/{count} ({100*accuracy/count:.1f}%)"
            )
        else:
            print(f"Prediction: {y_val:.4f} {real_or_fake(y)}")

        return result, accuracy, count, [y, y_val]
        
    except Exception as e:
        print(f"Error processing {vid}: {str(e)}")
        # Return default values on error
        result = store_result(
            result, os.path.basename(vid), 0, 0.5, klass, correct_label, compression
        )
        return result, accuracy, count, [0, 0.5]


def gen_parser():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser("GenConViT Updated Prediction")
    parser.add_argument("--p", type=str, help="video or directory path")
    parser.add_argument(
        "--f", type=int, default=15, help="number of frames to process for prediction"
    )
    parser.add_argument(
        "--d", type=str, default="other", help="dataset type: dfdc, faceforensics, timit, celeb, other"
    )
    parser.add_argument(
        "--s", help="model size type: tiny, large (currently not used in updated models)",
    )
    parser.add_argument(
        "--e", default='best_genconvit_ed_epoch_5', help="weight file name for ED model (without .pth extension)",
    )
    parser.add_argument(
        "--v", default='best_genconvit_vae_epoch_5', help="weight file name for VAE model (without .pth extension)",
    )
    parser.add_argument("--fp16", action="store_true", help="enable half precision support")
    parser.add_argument("--net", choices=['ed', 'vae', 'genconvit'], default='vae', 
                       help="network type: ed (encoder-decoder only), vae (VAE only), genconvit (ensemble)")

    args = parser.parse_args()
    
    # Validate arguments
    if not args.p:
        parser.error("--p (path) argument is required")
    
    path = args.p
    num_frames = args.f
    dataset = args.d
    fp16 = args.fp16
    net = args.net
    ed_weight = args.e
    vae_weight = args.v

    # Handle model size (legacy support)
    if args.s:
        if args.s in ['tiny', 'large']:
            config["model"]["backbone"] = f"convnext_{args.s}"
            config["model"]["embedder"] = f"swin_{args.s}_patch4_window7_224"
            config["model"]["type"] = args.s
            print(f"Note: Model size {args.s} specified, but may not be compatible with updated models")
    
    print(f'\nConfiguration:')
    print(f'  Network type: {net}')
    print(f'  ED weight: {ed_weight}')
    print(f'  VAE weight: {vae_weight}')
    print(f'  FP16: {fp16}')
    print(f'  Frames: {num_frames}')
    print(f'  Dataset: {dataset}')
    print()
    
    return path, dataset, num_frames, net, fp16, ed_weight, vae_weight


def main():
    """
    Main function
    """
    print("🚀 GenConViT Updated Prediction Script")
    print("=" * 50)
    
    start_time = perf_counter()
    
    try:
        path, dataset, num_frames, net, fp16, ed_weight, vae_weight = gen_parser()
        
        # Check if path exists
        if not os.path.exists(path):
            print(f"Error: Path {path} does not exist")
            return 1
        
        # Run prediction based on dataset type
        if dataset in ["dfdc", "faceforensics", "timit", "celeb"]:
            result = globals()[dataset](ed_weight, vae_weight, path, dataset, num_frames, net, fp16)
        else:
            result = vids(ed_weight, vae_weight, path, dataset, num_frames, net, fp16)

        # Save results
        curr_time = datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
        os.makedirs("result", exist_ok=True)
        file_path = os.path.join("result", f"prediction_{dataset}_{net}_{curr_time}.json")

        with open(file_path, "w") as f:
            json.dump(result, f, indent=2)
        
        # Print summary
        total_videos = len(result["video"]["name"])
        if total_videos > 0:
            fake_count = sum(1 for label in result["video"]["pred_label"] if label == "FAKE")
            real_count = total_videos - fake_count
            
            print(f"\n" + "="*50)
            print(f"PREDICTION SUMMARY")
            print(f"="*50)
            print(f"Total videos processed: {total_videos}")
            print(f"Predicted as REAL: {real_count}")
            print(f"Predicted as FAKE: {fake_count}")
            
            if "correct_label" in result["video"] and len(result["video"]["correct_label"]) > 0:
                correct_predictions = sum(1 for i in range(len(result["video"]["pred_label"])) 
                                        if result["video"]["pred_label"][i] == result["video"]["correct_label"][i])
                accuracy = correct_predictions / len(result["video"]["correct_label"]) * 100
                print(f"Accuracy: {correct_predictions}/{len(result['video']['correct_label'])} ({accuracy:.2f}%)")
            
            print(f"Results saved to: {file_path}")
        else:
            print("\nNo videos were successfully processed")
        
        end_time = perf_counter()
        print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
        print("✅ Prediction completed successfully!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Prediction interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())