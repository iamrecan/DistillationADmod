\
import torch
from PIL import Image
import os
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider # Added for slider
import cv2
from pathlib import Path

# Import specific trainers - Add more as needed
from models.SingleNet.trainer_sn import SnTrainer
# from models.DBFAD.trainer_dbfad import DbfadTrainer
# from models.EfficientAD.trainer_ead import EadTrainer
# from models.ReverseDistillation.trainer_rd import RdTrainer
# from models.StudentTeacher.trainer_st import StTrainer
from utils.functions import cal_anomaly_maps


# --- Configuration ---
DEFAULT_IMAGE_SIZE = (224, 224) # Should match the training image size for the model
INITIAL_THRESHOLD = 0.5

# --- Helper Functions ---

def load_trainer(model_type, config_dict, device):
    """Loads the specified model trainer."""
    print(f"Initializing trainer for model type: {model_type}")
    if model_type == "sn":
        return SnTrainer(config_dict, device)
    # Add other model types here as they are imported and handled
    # elif model_type == "dbfad":
    #     return DbfadTrainer(config_dict, device)
    # elif model_type == "ead":
    #     return EadTrainer(config_dict, device)
    # elif model_type == "rd":
    #     return RdTrainer(config_dict, device)
    # elif model_type == "st":
    #     return StTrainer(config_dict, device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Ensure trainer is imported and handled.")

def preprocess_image_for_trainer(image_path, image_size):
    """Loads and preprocesses an image for the trainer."""
    try:
        image = Image.open(image_path).convert("RGB")
        transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
        ])
        return transform(image).unsqueeze(0) # Add batch dimension
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def generate_blended_image(original_image_cv2, anomaly_map_numpy, current_threshold):
    """Generates the blended image with anomalies colored based on the threshold."""
    # Ensure anomaly map is HxW
    display_anomaly_map = anomaly_map_numpy.copy()
    if display_anomaly_map.ndim == 3 and display_anomaly_map.shape[0] == 1:
        display_anomaly_map = display_anomaly_map.squeeze(0)
    elif display_anomaly_map.ndim != 2:
        print(f"Error: Anomaly map has unexpected dimensions: {display_anomaly_map.shape}")
        # Return original image and error title if map is unusable
        return original_image_cv2, "Original Image (Anomaly Map Error)"

    h, w = original_image_cv2.shape[:2]
    try:
        anomaly_map_resized = cv2.resize(display_anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"Error resizing anomaly map: {e}. Displaying original image.")
        return original_image_cv2, "Original Image (Anomaly Map Resize Error)"

    overlay = np.zeros_like(original_image_cv2, dtype=np.uint8)
    highlight_color_anomaly = [0, 0, 255]  # Red for anomalies (BGR)
    highlight_color_normal = [0, 255, 0]   # Green for normal (BGR)

    above_threshold_mask = anomaly_map_resized > current_threshold
    below_threshold_mask = ~above_threshold_mask

    overlay[above_threshold_mask] = highlight_color_anomaly
    overlay[below_threshold_mask] = highlight_color_normal

    alpha = 0.4 # Transparency of the overlay
    blended_image = cv2.addWeighted(original_image_cv2, 1, overlay, alpha, 0)
    
    title = f"Anomalies Highlighted (Threshold: {current_threshold:.6f})"
    return blended_image, title

# --- Main Script ---
def main():
    print("Starting interactive inference V2 script...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Get Model Type and Weights Path from User ---
    model_type = input("Enter the model type (e.g., sn, dbfad, ead, rd, st): ").lower()
    model_weights_path_str = input("Enter the full path to the model weights (.pth file): ")
    model_weights_path = Path(model_weights_path_str)

    if not model_weights_path.exists() or not model_weights_path.is_file():
        print(f"Error: Model weights file not found or is not a file at '{model_weights_path_str}'. Exiting.")
        return

    # --- 2. Configure and Load Trainer ---
    # Minimal config for trainer initialization. Adjust if your trainer needs more.
    trainer_config = {
        "data_path": str(Path("./dataset")), # Placeholder, will be ignored in inference_only_mode
        "obj": "interactive_obj",            # Placeholder, will be ignored in inference_only_mode
        "save_path": str(Path("./results_interactive_v2")), # Placeholder
        "distillType": model_type,
        "inference_only_mode": True, # Instruct BaseTrainer to skip data loading etc.
        "TrainingData": {
            "img_size": DEFAULT_IMAGE_SIZE[0],
            "crop_size": DEFAULT_IMAGE_SIZE[0],
            "norm": True,
            "epochs": 1,  # Required by BaseTrainer.getParams, value not critical for inference
            "lr": 0.0001, # Required by BaseTrainer.getParams, value not critical for inference
            "batch_size": 1 # Required by BaseTrainer.getParams, value not critical for inference
        }
    }

    try:
        trainer = load_trainer(model_type, trainer_config, device)
        print(f"Attempting to load weights from: {model_weights_path}")
        
        state_dict = torch.load(model_weights_path, map_location=device)
        loaded_successfully = False
        target_model_to_load = trainer.model 
        potential_keys = ['model', 'model_state_dict', 'state_dict', 'student_state_dict', 'student', 'net', 'network']

        print(f"Successfully loaded file from {model_weights_path}.")
        print(f"Type of loaded data: {type(state_dict)}")
        if isinstance(state_dict, dict):
            print(f"Loaded data is a dictionary with keys: {list(state_dict.keys())}")

        if isinstance(state_dict, dict):
            for key in potential_keys:
                if key in state_dict:
                    print(f"Attempting to load weights from dictionary key: '{key}'...")
                    try:
                        target_model_to_load.load_state_dict(state_dict[key])
                        print(f"Successfully loaded weights using key '{key}'.")
                        loaded_successfully = True
                        break 
                    except Exception as e_load:
                        print(f"Failed to load weights from key '{key}': {e_load}")
            
            if not loaded_successfully:
                print("Could not load weights using common dictionary keys. Attempting to load the entire dictionary as state_dict...")
                try:
                    target_model_to_load.load_state_dict(state_dict)
                    print("Successfully loaded the entire dictionary as state_dict.")
                    loaded_successfully = True
                except Exception as e_direct_load:
                    print(f"Failed to load the entire dictionary directly as state_dict: {e_direct_load}")
        else:
            print("Loaded data is not a dictionary. Assuming it's the model's raw state_dict. Attempting to load...")
            try:
                target_model_to_load.load_state_dict(state_dict)
                print("Successfully loaded the raw state_dict.")
                loaded_successfully = True
            except Exception as e_raw_load:
                print(f"Failed to load the raw state_dict: {e_raw_load}")

        if not loaded_successfully:
            error_message_parts = [
                f"Critical: Could not load model weights into 'trainer.model' from '{model_weights_path}'.",
                f"The loaded file was of type: {type(state_dict)}."
            ]
            if isinstance(state_dict, dict):
                error_message_parts.append(f"If it was a dictionary, its keys were: {list(state_dict.keys())}.")
            error_message_parts.extend([
                "Please ensure that:",
                f"1. The .pth file is a valid PyTorch state_dictionary or a dictionary containing it under a compatible key (tried: {potential_keys}, or direct load of the dictionary/object).",
                "2. The architecture of the model in the trainer (assumed to be 'trainer.model') matches the saved weights (check for layer name mismatches, different layer sizes, etc.).",
                "3. The trainer (e.g., 'SnTrainer') correctly defines 'self.model' as the network to be loaded, or adjust 'target_model_to_load' in this script accordingly."
            ])
            raise RuntimeError("\n".join(error_message_parts))

        print("Model weights loaded successfully into trainer.model.")
        target_model_to_load.eval() 
    
    except Exception as e:
        print(f"Error during trainer initialization or model weight loading: {e}")
        print("\nDetailed context for failure:")
        print(f"  Model Type: {model_type}")
        print(f"  Weights Path: {model_weights_path}")
        print("  Common issues to check:")
        print("    - Is the model_type correct and handled in the 'load_trainer' function?")
        print("    - Is the .pth file path correct, and is the file a valid, uncorrupted PyTorch weights file?")
        print("    - Are the weights compatible with the model architecture defined in the selected trainer?")
        print("      (e.g., mismatches in layer names, sizes, or structure).")
        print("    - If the .pth file is a dictionary, does it contain the state_dict under an expected key, or is the entire dictionary the state_dict?")
        print("    - Does the trainer's 'self.model' attribute correctly point to the PyTorch nn.Module to be loaded?")
        print("Review the messages above for specific errors encountered during the loading process.")
        return

    # --- 3. Get Image Path from User ---
    while True:
        image_path_str = input("Enter the path to the image for anomaly detection: ")
        image_path = Path(image_path_str)
        if image_path.exists() and image_path.is_file():
            break
        else:
            print(f"Error: Image not found or is not a file at '{image_path_str}'. Please try again.")
    
    print(f"Using image: {image_path}")

    # --- 4. Load and Preprocess Image ---
    input_tensor = preprocess_image_for_trainer(str(image_path), DEFAULT_IMAGE_SIZE)
    if input_tensor is None:
        print("Failed to load or preprocess the image. Exiting.")
        return
    input_tensor = input_tensor.to(device)

    try:
        original_image_cv2 = cv2.imread(str(image_path))
        if original_image_cv2 is None:
            raise ValueError("cv2.imread returned None. Check image path and integrity.")
    except Exception as e:
        print(f"Error loading original image with OpenCV from {image_path}: {e}")
        return

    # --- 5. Perform Inference to get Anomaly Map ---
    print("Performing inference to generate anomaly map...")
    try:
        with torch.no_grad():
            trainer.infer(input_tensor) 
            trainer.post_process() 
            
            if not hasattr(trainer, 'features_s') or not hasattr(trainer, 'features_t'):
                raise AttributeError("Trainer object is missing 'features_s' or 'features_t' attributes after infer/post_process.")
            if not hasattr(trainer, 'img_cropsize') or not hasattr(trainer, 'norm'): # Ensure these are set from config
                # Attempt to get them from trainer_config if not on trainer directly (should be set by getParams)
                trainer.img_cropsize = trainer_config['TrainingData']['crop_size']
                trainer.norm = trainer_config['TrainingData']['norm']
                # raise AttributeError("Trainer object is missing 'img_cropsize' or 'norm' attributes from config.")


            anomaly_map_raw = cal_anomaly_maps(
                trainer.features_s,
                trainer.features_t,
                out_size=trainer.img_cropsize, 
                norm=trainer.norm
            )
        
        if isinstance(anomaly_map_raw, torch.Tensor):
            if anomaly_map_raw.ndim == 4 and anomaly_map_raw.shape[0] == 1 and anomaly_map_raw.shape[1] == 1:
                anomaly_map_numpy = anomaly_map_raw.squeeze(0).squeeze(0).cpu().numpy()
            elif anomaly_map_raw.ndim == 3 and anomaly_map_raw.shape[0] == 1:
                anomaly_map_numpy = anomaly_map_raw.squeeze(0).cpu().numpy()
            elif anomaly_map_raw.ndim == 2:
                anomaly_map_numpy = anomaly_map_raw.cpu().numpy()
            else:
                print(f"Unexpected anomaly map tensor shape: {anomaly_map_raw.shape}. Attempting to use first element if list, else direct.")
                anomaly_map_numpy = anomaly_map_raw[0].cpu().numpy() if isinstance(anomaly_map_raw, list) else anomaly_map_raw.cpu().numpy()
        elif isinstance(anomaly_map_raw, np.ndarray):
            if anomaly_map_raw.ndim == 4 and anomaly_map_raw.shape[0] == 1 and anomaly_map_raw.shape[1] == 1:
                anomaly_map_numpy = anomaly_map_raw.squeeze(0).squeeze(0)
            elif anomaly_map_raw.ndim == 3 and anomaly_map_raw.shape[0] == 1:
                anomaly_map_numpy = anomaly_map_raw.squeeze(0)
            elif anomaly_map_raw.ndim == 2:
                anomaly_map_numpy = anomaly_map_raw
            else:
                print(f"Unexpected anomaly map numpy array shape: {anomaly_map_raw.shape}. Assuming it's (H,W) or needs specific handling.")
                anomaly_map_numpy = anomaly_map_raw
        else:
            raise TypeError(f"Anomaly map type not recognized: {type(anomaly_map_raw)}")

        print("Inference complete. Anomaly map generated.")
    except Exception as e:
        print(f"Error during model inference or post-processing: {e}")
        print("This could be due to an issue with the model architecture, input tensor shape, or trainer methods.")
        return

    # --- 6. Interactive Thresholding with Slider ---
    print("\n--- Interactive Threshold Adjustment with Slider ---")
    
    fig_suptitle_text = f"File: {image_path.name} | Model: {model_type.upper()}"

    # --- Robust Slider Setup ---
    # Check for and handle NaNs/Infs in the anomaly map
    if anomaly_map_numpy is None or anomaly_map_numpy.size == 0:
        print("Error: Anomaly map is None or empty. Cannot proceed with interactive thresholding.")
        return
        
    if not np.all(np.isfinite(anomaly_map_numpy)):
        print("Warning: Anomaly map contains NaNs or Infs. These will be replaced.")
        # Replacing NaNs with a value that represents low anomaly (e.g., min of finite values, or 0)
        # Replacing Infs with a value that represents high anomaly (e.g., max of finite values, or 1)
        # This strategy depends on how your anomaly scores are scaled.
        # If scores are typically 0-1, nan=0, posinf=1, neginf=0 is reasonable.
        finite_min = np.min(anomaly_map_numpy[np.isfinite(anomaly_map_numpy)]) if np.any(np.isfinite(anomaly_map_numpy)) else 0.0
        finite_max = np.max(anomaly_map_numpy[np.isfinite(anomaly_map_numpy)]) if np.any(np.isfinite(anomaly_map_numpy)) else 1.0
        anomaly_map_numpy = np.nan_to_num(anomaly_map_numpy, nan=finite_min, posinf=finite_max, neginf=finite_min)

    min_val = float(np.min(anomaly_map_numpy))
    max_val = float(np.max(anomaly_map_numpy))

    if min_val >= max_val: 
        print(f"Warning: Anomaly map range is invalid (min: {min_val}, max: {max_val}). Using default slider range [0,1] or adjusted if min_val is the only value.")
        if min_val == max_val: # Map is completely flat
            # To make the slider movable, we need a range. 
            # If the flat value is e.g. 0.5, we can set range [0,1]. If it's far from [0,1], adjust.
            if 0.0 <= min_val <= 1.0:
                max_val = min_val + 0.1 # Create a small range if min_val is already sensible
                if max_val > 1.0: max_val = 1.0
                if min_val == max_val : min_val = max_val - 0.1 # if max_val was capped at 1.0 and min_val was 1.0
                if min_val < 0.0: min_val = 0.0
            else: # If flat value is outside typical 0-1, center range around it or use fixed [0,1]
                min_val = 0.0
                max_val = 1.0
        else: # Should not happen if min_val >= max_val, but as a fallback
            min_val = 0.0
            max_val = 1.0 
    
    # Ensure min_val is strictly less than max_val after adjustments
    if min_val >= max_val:
        min_val = 0.0
        max_val = 1.0
        print(f"Fallback: Setting slider range to [{min_val}, {max_val}]")

    current_threshold_val = float(np.clip(INITIAL_THRESHOLD, min_val, max_val))
    
    # Define valstep
    if abs(max_val - min_val) < 1e-6: 
        val_step = 1e-3 
    else:
        val_step = (max_val - min_val) / 100.0
    
    if val_step <= 0: # Final check for val_step
        val_step = 0.001 

    # Initial blended image and title using the potentially adjusted current_threshold_val
    initial_blended_image, initial_ax_title = generate_blended_image(
        original_image_cv2, anomaly_map_numpy, current_threshold_val
    )

    fig, ax = plt.subplots(figsize=(10, 9)) # Adjusted figsize for slider
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9) # Make space for slider and suptitle

    fig.suptitle(fig_suptitle_text)

    img_display = ax.imshow(cv2.cvtColor(initial_blended_image, cv2.COLOR_BGR2RGB))
    ax.set_title(initial_ax_title)
    ax.axis("off")

    # Define slider axis position and color
    ax_slider_rect = [0.20, 0.05, 0.60, 0.03] # x, y, width, height from bottom left
    ax_slider = fig.add_axes(ax_slider_rect, facecolor='lightgoldenrodyellow')
    
    threshold_slider = Slider(
        ax=ax_slider,
        label='Threshold',
        valmin=min_val,
        valmax=max_val,
        valinit=current_threshold_val,
        valstep=val_step 
    )

    def update_plot(val):
        new_threshold = threshold_slider.val # Get current value from slider
        blended_img, new_ax_title = generate_blended_image(
            original_image_cv2, anomaly_map_numpy, new_threshold
        )
        img_display.set_data(cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB))
        ax.set_title(new_ax_title)
        fig.canvas.draw_idle() # Redraw the figure

    threshold_slider.on_changed(update_plot)
    
    plt.show() # This will block until the window is closed.
    
    print("Interactive session finished.")
    print("Script finished.")

if __name__ == "__main__":
    main()
