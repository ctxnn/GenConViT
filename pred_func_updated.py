import os
import numpy as np
import cv2
import torch
import dlib
import face_recognition
from torchvision import transforms
from tqdm import tqdm
from dataset.loader import normalize_data
from model.config import load_config
from genconvit_updated import GenConViT
from decord import VideoReader, cpu

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_genconvit(config, net, ed_weight, vae_weight, fp16):
    """
    Load the updated GenConViT model that works with new architectures
    
    Args:
        config: Model configuration dictionary
        net: Network type ('ed', 'vae', or 'genconvit')
        ed_weight: Path to ED model weights
        vae_weight: Path to VAE model weights
        fp16: Whether to use half precision
    
    Returns:
        Loaded model
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_str}")
    
    try:
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

        # Print model info
        model_info = model.get_model_info()
        print("Model Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")

        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def face_rec(frames, p=None, klass=None):
    """
    Extract faces from frames using face_recognition library
    Resize faces to 128x128 to match new model architecture
    """
    temp_face = np.zeros((len(frames), 128, 128, 3), dtype=np.uint8)  # Changed to 128x128
    count = 0
    mod = "cnn" if dlib.DLIB_USE_CUDA else "hog"

    for _, frame in tqdm(enumerate(frames), total=len(frames), desc="Extracting faces"):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        face_locations = face_recognition.face_locations(
            frame, number_of_times_to_upsample=0, model=mod
        )

        for face_location in face_locations:
            if count < len(frames):
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]
                # Resize to 128x128 instead of 224x224
                face_image = cv2.resize(
                    face_image, (128, 128), interpolation=cv2.INTER_AREA
                )
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

                temp_face[count] = face_image
                count += 1
            else:
                break

    return ([], 0) if count == 0 else (temp_face[:count], count)


def preprocess_frame(frame):
    """
    Preprocess frames for the updated model
    """
    if len(frame) == 0:
        return torch.empty(0)
        
    # Create tensor on CPU first to avoid memory issues
    df_tensor = torch.tensor(frame).float()
    df_tensor = df_tensor.permute((0, 3, 1, 2))

    # Normalize the data
    try:
        normalizer = normalize_data()["vid"]
        for i in range(len(df_tensor)):
            df_tensor[i] = normalizer(df_tensor[i] / 255.0)
    except Exception as e:
        print(f"Warning: Normalization failed, using simple normalization: {e}")
        # Fallback normalization
        df_tensor = df_tensor / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        df_tensor = (df_tensor - mean) / std

    # Only move to GPU if available after preprocessing
    if torch.cuda.is_available():
        df_tensor = df_tensor.to(device)

    return df_tensor


def pred_vid(df, model):
    """
    Make prediction on video frames using the updated model
    """
    if len(df) == 0:
        return 0, 0.5  # Default values if no faces detected
        
    with torch.no_grad():
        try:
            # Get the device of the model parameters
            model_device = next(model.parameters()).device
            
            # Ensure input tensor is on the same device as the model
            if df.device != model_device:
                df = df.to(model_device)
            
            # Forward pass through the model
            output = model(df)
            
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(output)
            
            return max_prediction_value(probabilities)
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            print(f"Input shape: {df.shape if hasattr(df, 'shape') else 'Unknown'}")
            return 0, 0.5  # Return default values on error


def max_prediction_value(y_pred):
    """
    Finds the index and value of the maximum prediction value.
    """
    if len(y_pred.shape) == 1:
        # Single prediction
        mean_val = y_pred
    else:
        # Multiple predictions, take mean
        mean_val = torch.mean(y_pred, dim=0)
    
    # Ensure we have 2 classes
    if len(mean_val) == 1:
        # Single output, assume it's probability of fake
        fake_prob = mean_val[0].item()
        real_prob = 1 - fake_prob
        mean_val = torch.tensor([real_prob, fake_prob])
    
    prediction_idx = torch.argmax(mean_val).item()
    confidence = mean_val[prediction_idx].item()
    
    return prediction_idx, confidence


def real_or_fake(prediction):
    """
    Convert prediction index to human readable format
    """
    return {0: "REAL", 1: "FAKE"}[prediction]


def extract_frames(video_file, frames_nums=15):
    """
    Extract frames from video file
    """
    try:
        vr = VideoReader(video_file, ctx=cpu(0))
        total_frames = len(vr)
        
        if total_frames == 0:
            return np.array([])
            
        step_size = max(1, total_frames // frames_nums)
        frame_indices = list(range(0, total_frames, step_size))[:frames_nums]
        
        frames = vr.get_batch(frame_indices).asnumpy()
        return frames
        
    except Exception as e:
        print(f"Error extracting frames from {video_file}: {e}")
        return np.array([])


def df_face(vid, num_frames, net):
    """
    Extract faces from video for deepfake detection
    """
    try:
        img = extract_frames(vid, num_frames)
        
        if len(img) == 0:
            print(f"No frames extracted from {vid}")
            return torch.empty(0)
            
        face, count = face_rec(img)
        
        if count == 0:
            print(f"No faces detected in {vid}")
            return torch.empty(0)
            
        return preprocess_frame(face)
        
    except Exception as e:
        print(f"Error processing video {vid}: {e}")
        return torch.empty(0)


def is_video(vid):
    """
    Check if file is a valid video file
    """
    if not os.path.isfile(vid):
        return False
        
    video_extensions = [".avi", ".mp4", ".mpg", ".mpeg", ".mov", ".mkv", ".webm", ".flv"]
    return vid.lower().endswith(tuple(video_extensions))


def set_result():
    """
    Initialize result dictionary structure
    """
    return {
        "video": {
            "name": [],
            "pred": [],
            "klass": [],
            "pred_label": [],
            "correct_label": [],
        }
    }


def store_result(result, filename, y, y_val, klass, correct_label=None, compression=None):
    """
    Store prediction results
    """
    result["video"]["name"].append(filename)
    result["video"]["pred"].append(y_val)
    result["video"]["klass"].append(klass.lower())
    result["video"]["pred_label"].append(real_or_fake(y))

    if correct_label is not None:
        result["video"]["correct_label"].append(correct_label)

    if compression is not None:
        if "compression" not in result["video"]:
            result["video"]["compression"] = []
        result["video"]["compression"].append(compression)

    return result


def batch_predict(video_paths, model, num_frames=15, net='genconvit'):
    """
    Batch prediction for multiple videos
    """
    results = []
    
    for video_path in tqdm(video_paths, desc="Processing videos"):
        if not is_video(video_path):
            print(f"Skipping non-video file: {video_path}")
            continue
            
        try:
            df = df_face(video_path, num_frames, net)
            
            if len(df) == 0:
                print(f"No faces detected in {video_path}")
                results.append({
                    'video': os.path.basename(video_path),
                    'prediction': 0,
                    'confidence': 0.5,
                    'label': 'REAL',
                    'error': 'No faces detected'
                })
                continue
                
            pred_idx, confidence = pred_vid(df, model)
            
            results.append({
                'video': os.path.basename(video_path),
                'prediction': pred_idx,
                'confidence': confidence,
                'label': real_or_fake(pred_idx),
                'error': None
            })
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            results.append({
                'video': os.path.basename(video_path),
                'prediction': 0,
                'confidence': 0.5,
                'label': 'REAL',
                'error': str(e)
            })
    
    return results


def get_model_summary(model):
    """
    Get summary information about the loaded model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = {
        'total_parameters': f"{total_params:,}",
        'trainable_parameters': f"{trainable_params:,}",
        'model_type': model.net,
        'fp16_enabled': model.fp16,
        'device': next(model.parameters()).device.type
    }
    
    return summary