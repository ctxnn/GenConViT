import os
import numpy as np
import cv2
import torch
import dlib
import face_recognition
from torchvision import transforms
from tqdm import tqdm
from dataset.loader import normalize_data
from .config import load_config
from .genconvit import GenConViT
from .genconvit_v2 import GenConViTV2
from decord import VideoReader, cpu

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_genconvit(config, net, ed_weight, vae_weight, fp16, arch_type='original', use_attention=True, use_residual=True):
    """
    Load either the original GenConViT model or the enhanced V2 architecture
    
    Args:
        config: Model configuration dictionary
        net: Network type ('ed', 'vae', 'genconvit', or 'v2')
        ed_weight: Path to ED model weights
        vae_weight: Path to VAE model weights
        fp16: Whether to use half precision
        arch_type: Architecture type ('original' or 'v2')
        use_attention: Whether to use attention mechanism (for V2 only)
        use_residual: Whether to use residual connections (for V2 only)
    
    Returns:
        Loaded model
    """
    # Explicitly set the device to use
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_str}")
    
    if arch_type == 'v2':
        model = GenConViTV2(
            config,
            ed=ed_weight,
            vae=vae_weight,
            net=net,
            fp16=fp16,
            use_attention=use_attention,
            use_residual=use_residual
        )
    else:
        model = GenConViT(
            config,
            ed=ed_weight,
            vae=vae_weight,
            net=net,
            fp16=fp16
        )

    # Move model to device
    model.to(device)
    model.eval()
    if fp16:
        model.half()

    return model


def face_rec(frames, p=None, klass=None):
    temp_face = np.zeros((len(frames), 224, 224, 3), dtype=np.uint8)
    count = 0
    mod = "cnn" if dlib.DLIB_USE_CUDA else "hog"

    for _, frame in tqdm(enumerate(frames), total=len(frames)):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        face_locations = face_recognition.face_locations(
            frame, number_of_times_to_upsample=0, model=mod
        )

        for face_location in face_locations:
            if count < len(frames):
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]
                face_image = cv2.resize(
                    face_image, (224, 224), interpolation=cv2.INTER_AREA
                )
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

                temp_face[count] = face_image
                count += 1
            else:
                break

    return ([], 0) if count == 0 else (temp_face[:count], count)


def preprocess_frame(frame):
    # Create tensor on CPU first to avoid memory issues
    df_tensor = torch.tensor(frame).float()
    df_tensor = df_tensor.permute((0, 3, 1, 2))

    # Normalize the data
    for i in range(len(df_tensor)):
        df_tensor[i] = normalize_data()["vid"](df_tensor[i] / 255.0)

    # Only move to GPU if available after preprocessing
    if torch.cuda.is_available():
        df_tensor = df_tensor.to(device)

    return df_tensor


def pred_vid(df, model):
    with torch.no_grad():
        # Get the device of the model parameters
        model_device = next(model.parameters()).device
        
        # Ensure input tensor is on the same device as the model
        if df.device != model_device:
            df = df.to(model_device)
            
        return max_prediction_value(torch.sigmoid(model(df).squeeze()))


def max_prediction_value(y_pred):
    # Finds the index and value of the maximum prediction value.
    mean_val = torch.mean(y_pred, dim=0)
    return (
        torch.argmax(mean_val).item(),
        mean_val[0].item()
        if mean_val[0] > mean_val[1]
        else abs(1 - mean_val[1]).item(),
    )


def real_or_fake(prediction):
    return {0: "REAL", 1: "FAKE"}[prediction ^ 1]


def extract_frames(video_file, frames_nums=15):
    vr = VideoReader(video_file, ctx=cpu(0))
    step_size = max(1, len(vr) // frames_nums)  # Calculate the step size between frames
    return vr.get_batch(
        list(range(0, len(vr), step_size))[:frames_nums]
    ).asnumpy()  # seek frames with step_size


def df_face(vid, num_frames, net):
    img = extract_frames(vid, num_frames)
    face, count = face_rec(img)
    return preprocess_frame(face) if count > 0 else []


def is_video(vid):
    return os.path.isfile(vid) and vid.endswith(
        tuple([".avi", ".mp4", ".mpg", ".mpeg", ".mov"])
    )


def set_result():
    return {
        "video": {
            "name": [],
            "pred": [],
            "klass": [],
            "pred_label": [],
            "correct_label": [],
        }
    }


def store_result(
    result, filename, y, y_val, klass, correct_label=None, compression=None
):
    result["video"]["name"].append(filename)
    result["video"]["pred"].append(y_val)
    result["video"]["klass"].append(klass.lower())
    result["video"]["pred_label"].append(real_or_fake(y))

    if correct_label is not None:
        result["video"]["correct_label"].append(correct_label)

    if compression is not None:
        result["video"]["compression"].append(compression)

    return result
