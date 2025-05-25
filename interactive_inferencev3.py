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

# --- Adaptive Threshold Functions ---

def calculate_adaptive_threshold(anomaly_map, method='combined'):
    """
    Calculate adaptive threshold based on anomaly map statistics.
    
    Args:
        anomaly_map: numpy array of anomaly scores
        method: 'percentile', 'statistical', 'iqr', or 'combined'
        
    Returns:
        float: calculated threshold value
    """
    if anomaly_map is None or anomaly_map.size == 0:
        print("Warning: Empty anomaly map, using default threshold")
        return INITIAL_THRESHOLD
    
    # Clean the data
    anomaly_map_clean = anomaly_map[np.isfinite(anomaly_map)]
    if anomaly_map_clean.size == 0:
        print("Warning: No finite values in anomaly map, using default threshold")
        return INITIAL_THRESHOLD
    
    if method == 'percentile':
        # Use 95th percentile as threshold
        threshold = np.percentile(anomaly_map_clean, 95)
    
    elif method == 'statistical':
        # Use mean + 2.5 * std as threshold
        mean_val = np.mean(anomaly_map_clean)
        std_val = np.std(anomaly_map_clean)
        threshold = mean_val + 2.5 * std_val
    
    elif method == 'iqr':
        # Use IQR-based outlier detection
        q1 = np.percentile(anomaly_map_clean, 25)
        q3 = np.percentile(anomaly_map_clean, 75)
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr
    
    elif method == 'combined':
        # Combine all methods and take the median
        percentile_thresh = np.percentile(anomaly_map_clean, 95)
        
        mean_val = np.mean(anomaly_map_clean)
        std_val = np.std(anomaly_map_clean)
        statistical_thresh = mean_val + 2.5 * std_val
        
        q1 = np.percentile(anomaly_map_clean, 25)
        q3 = np.percentile(anomaly_map_clean, 75)
        iqr = q3 - q1
        iqr_thresh = q3 + 1.5 * iqr
        
        # Take median of the three methods
        thresholds = [percentile_thresh, statistical_thresh, iqr_thresh]
        threshold = np.median(thresholds)
        
        print(f"Adaptive threshold calculation:")
        print(f"  Percentile (95th): {percentile_thresh:.6f}")
        print(f"  Statistical (μ+2.5σ): {statistical_thresh:.6f}")
        print(f"  IQR-based (Q3+1.5*IQR): {iqr_thresh:.6f}")
        print(f"  Combined (median): {threshold:.6f}")
    
    else:
        print(f"Unknown method '{method}', using percentile method")
        threshold = np.percentile(anomaly_map_clean, 95)
    
    # Ensure threshold is within reasonable bounds
    min_val = np.min(anomaly_map_clean)
    max_val = np.max(anomaly_map_clean)
    threshold = np.clip(threshold, min_val, max_val)
    
    return float(threshold)

def analyze_anomaly_distribution(anomaly_map):
    """
    Analyze the distribution of anomaly scores for better threshold selection.
    
    Args:
        anomaly_map: numpy array of anomaly scores
        
    Returns:
        dict: statistics about the anomaly distribution
    """
    if anomaly_map is None or anomaly_map.size == 0:
        return {}
    
    anomaly_map_clean = anomaly_map[np.isfinite(anomaly_map)]
    if anomaly_map_clean.size == 0:
        return {}
    
    stats = {
        'min': np.min(anomaly_map_clean),
        'max': np.max(anomaly_map_clean),
        'mean': np.mean(anomaly_map_clean),
        'std': np.std(anomaly_map_clean),
        'median': np.median(anomaly_map_clean),
        'q1': np.percentile(anomaly_map_clean, 25),
        'q3': np.percentile(anomaly_map_clean, 75),
        'p95': np.percentile(anomaly_map_clean, 95),
        'p99': np.percentile(anomaly_map_clean, 99)
    }
    
    # Calculate potential anomaly percentage with different thresholds
    for percentile in [90, 95, 99]:
        threshold = np.percentile(anomaly_map_clean, percentile)
        anomaly_count = np.sum(anomaly_map_clean > threshold)
        total_pixels = anomaly_map_clean.size
        anomaly_percentage = (anomaly_count / total_pixels) * 100
        stats[f'anomaly_pct_p{percentile}'] = anomaly_percentage
    
    return stats

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
        
        # --- 5.1. Calculate Adaptive Threshold ---
        print("\n--- Calculating Adaptive Threshold ---")
        
        # Analyze anomaly distribution
        distribution_stats = analyze_anomaly_distribution(anomaly_map_numpy)
        if distribution_stats:
            print(f"Anomaly map statistics:")
            print(f"  Range: [{distribution_stats['min']:.6f}, {distribution_stats['max']:.6f}]")
            print(f"  Mean: {distribution_stats['mean']:.6f}, Std: {distribution_stats['std']:.6f}")
            print(f"  Median: {distribution_stats['median']:.6f}")
            print(f"  Quartiles: Q1={distribution_stats['q1']:.6f}, Q3={distribution_stats['q3']:.6f}")
            print(f"  Percentiles: P95={distribution_stats['p95']:.6f}, P99={distribution_stats['p99']:.6f}")
            
            # Show potential anomaly percentages
            for percentile in [90, 95, 99]:
                pct_key = f'anomaly_pct_p{percentile}'
                if pct_key in distribution_stats:
                    print(f"  Anomaly % at P{percentile}: {distribution_stats[pct_key]:.2f}%")
        
        # Calculate adaptive threshold using combined method
        adaptive_threshold = calculate_adaptive_threshold(anomaly_map_numpy, method='combined')
        print(f"\nRecommended adaptive threshold: {adaptive_threshold:.6f}")
        
        # Ask user for threshold choice
        print(f"\nThreshold options:")
        print(f"1. Use adaptive threshold: {adaptive_threshold:.6f}")
        print(f"2. Use default threshold: {INITIAL_THRESHOLD:.6f}")
        print(f"3. Enter custom threshold")
        
        choice = input("Choose threshold option (1/2/3): ").strip()
        
        if choice == '1':
            current_threshold_val = adaptive_threshold
            print(f"Using adaptive threshold: {current_threshold_val:.6f}")
        elif choice == '2':
            current_threshold_val = INITIAL_THRESHOLD
            print(f"Using default threshold: {current_threshold_val:.6f}")
        elif choice == '3':
            try:
                custom_threshold = float(input("Enter custom threshold value: "))
                current_threshold_val = custom_threshold
                print(f"Using custom threshold: {current_threshold_val:.6f}")
            except ValueError:
                print("Invalid input, using adaptive threshold as fallback")
                current_threshold_val = adaptive_threshold
        else:
            print("Invalid choice, using adaptive threshold as default")
            current_threshold_val = adaptive_threshold
            
    except Exception as e:
        print(f"Error during model inference or post-processing: {e}")
        print("This could be due to an issue with the model architecture, input tensor shape, or trainer methods.")
        return

    # --- 5. Threshold Selection with Adaptive System ---
    print("\n--- Threshold Selection ---")
    
    # Calculate adaptive thresholds
    adaptive_thresholds = calculate_adaptive_thresholds(anomaly_map_numpy)
    print("\nAdaptive Threshold Analysis:")
    print(f"Otsu threshold: {adaptive_thresholds['otsu']:.6f}")
    print(f"Mean + 2*Std: {adaptive_thresholds['statistical']:.6f}")
    print(f"95th percentile: {adaptive_thresholds['percentile_95']:.6f}")
    print(f"99th percentile: {adaptive_thresholds['percentile_99']:.6f}")
    
    # Get recommended threshold
    recommended_threshold = get_recommended_threshold(anomaly_map_numpy)
    print(f"\nRecommended threshold: {recommended_threshold:.6f}")
    
    # Interactive threshold selection
    print("\nThreshold Selection Options:")
    print("1. Use recommended adaptive threshold")
    print("2. Use Otsu threshold")
    print("3. Use statistical threshold (mean + 2*std)")
    print("4. Use 95th percentile")
    print("5. Use 99th percentile")
    print("6. Enter custom threshold")
    
    choice = input("Enter your choice (1-6) [default: 1]: ").strip()
    
    if choice == "2":
        current_threshold_val = adaptive_thresholds['otsu']
        threshold_method = "Otsu"
    elif choice == "3":
        current_threshold_val = adaptive_thresholds['statistical']
        threshold_method = "Statistical"
    elif choice == "4":
        current_threshold_val = adaptive_thresholds['percentile_95']
        threshold_method = "95th Percentile"
    elif choice == "5":
        current_threshold_val = adaptive_thresholds['percentile_99']
        threshold_method = "99th Percentile"
    elif choice == "6":
        try:
            current_threshold_val = float(input("Enter threshold value: "))
            threshold_method = "Custom"
        except ValueError:
            print("Invalid input. Using recommended threshold.")
            current_threshold_val = recommended_threshold
            threshold_method = "Recommended (default)"
    else:  # Default to recommended
        current_threshold_val = recommended_threshold
        threshold_method = "Recommended"
    
    print(f"\nSelected threshold: {current_threshold_val:.6f} ({threshold_method})")
    
    # Preview the threshold effect
    binary_preview = (anomaly_map_numpy > current_threshold_val).astype(np.uint8)
    anomaly_percentage_preview = (np.sum(binary_preview) / binary_preview.size) * 100
    print(f"Preview: {anomaly_percentage_preview:.2f}% of pixels will be marked as anomalous")

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

    # --- 7. Extract and Prepare Points for SAM based on Final Threshold ---
    final_threshold = threshold_slider.val
    print(f"Final threshold selected: {final_threshold}")

    # Ensure anomaly_map_numpy and original_image_cv2 are available here
    # anomaly_map_numpy is (H_map, W_map)
    # original_image_cv2 is (H_orig, W_orig, C)

    if anomaly_map_numpy is not None and original_image_cv2 is not None:
        H_map, W_map = anomaly_map_numpy.shape[:2] # Assuming anomaly_map_numpy is 2D or 3D (1,H,W)
        if anomaly_map_numpy.ndim == 3 and anomaly_map_numpy.shape[0] == 1: # Handle case where it might be (1,H,W)
            anomaly_map_numpy_2d = anomaly_map_numpy.squeeze(0)
            H_map, W_map = anomaly_map_numpy_2d.shape
        elif anomaly_map_numpy.ndim == 2:
            anomaly_map_numpy_2d = anomaly_map_numpy
        else:
            print(f"Error: Anomaly map has unexpected dimensions for point extraction: {anomaly_map_numpy.shape}")
            anomaly_map_numpy_2d = None

        if anomaly_map_numpy_2d is not None:
            H_orig, W_orig = original_image_cv2.shape[:2]

            # Create binary mask using the final threshold
            final_mask = anomaly_map_numpy_2d > final_threshold
            
            # Get coordinates (row, col) which are (y_map, x_map)
            points_yx_map = np.argwhere(final_mask)

            sam_input_points = []
            if points_yx_map.size > 0:
                scale_x = W_orig / W_map
                scale_y = H_orig / H_map
                
                for y_m, x_m in points_yx_map:
                    x_orig = int(x_m * scale_x)
                    y_orig = int(y_m * scale_y)
                    sam_input_points.append([x_orig, y_orig])
                
                print(f"\\n--- Input Points for SAM (scaled to original image size {W_orig}x{H_orig}) ---")
                print(f"Found {len(sam_input_points)} points above threshold {final_threshold:.6f}.")
                # print("Coordinates (x, y):")
                # for pt in sam_input_points:
                #     print(pt)
                # To keep the output cleaner, you might want to save these points to a file
                # or pass them directly to your SAM script.
                # Example: print(f"SAM points: {sam_input_points}") 
                # This list can be very long, so printing all points might not be ideal.
                # Consider saving to a file or further processing.
                output_points_filepath = Path(image_path.parent) / f"{image_path.stem}_sam_points_thresh_{final_threshold:.4f}.txt"
                try:
                    with open(output_points_filepath, 'w') as f:
                        for pt in sam_input_points:
                            f.write(f"{pt[0]},{pt[1]}\\n")
                    print(f"Anomaly point coordinates saved to: {output_points_filepath}")
                except Exception as e_save:
                    print(f"Error saving points to file: {e_save}")
                
                print(f"\\nThese points can be used as input for a SAM model, for example, with '{Path('segment_with_sam2/segment_anomaly_sam.py')}'")
                print("Each point is in [x, y] format for the original image.")
            else:
                print(f"\\nNo points found above threshold {final_threshold:.6f}.")
        else:
            print("\\nCould not extract points due to anomaly map dimension issues.")
    else:
        print("\\nAnomaly map or original image not available for point extraction.")

    print("Script finished.")

    # --- 6. Visualization with Adaptive Threshold ---
    print("\n--- Creating Visualization ---")
    
    # Apply the selected threshold
    binary_mask = (anomaly_map_numpy > current_threshold_val).astype(np.uint8)
    anomaly_percentage = (np.sum(binary_mask) / binary_mask.size) * 100
    
    # Create overlay
    overlay_img = input_image.copy()
    anomaly_overlay = np.zeros_like(input_image)
    anomaly_overlay[binary_mask == 1] = [0, 0, 255]  # Red for anomalies
    overlay_img = cv2.addWeighted(overlay_img, 0.7, anomaly_overlay, 0.3, 0)
    
    # Create visualization plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Anomaly Detection Results - {threshold_method} Threshold ({current_threshold_val:.6f})', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Anomaly map (heatmap)
    im1 = axes[0, 1].imshow(anomaly_map_numpy, cmap='hot', interpolation='nearest')
    axes[0, 1].set_title('Anomaly Heatmap')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    
    # Binary mask
    axes[0, 2].imshow(binary_mask, cmap='gray')
    axes[0, 2].set_title(f'Binary Mask\n({anomaly_percentage:.2f}% anomalous)')
    axes[0, 2].axis('off')
    
    # Overlay visualization
    axes[1, 0].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Anomaly Overlay')
    axes[1, 0].axis('off')
    
    # Threshold comparison chart
    thresholds_to_plot = ['otsu', 'statistical', 'percentile_95', 'percentile_99']
    threshold_values = [adaptive_thresholds[t] for t in thresholds_to_plot]
    threshold_labels = ['Otsu', 'Statistical', '95th %ile', '99th %ile']
    
    bars = axes[1, 1].bar(threshold_labels, threshold_values, alpha=0.7)
    axes[1, 1].axhline(y=current_threshold_val, color='red', linestyle='--', 
                       label=f'Selected: {threshold_method}')
    axes[1, 1].set_title('Threshold Comparison')
    axes[1, 1].set_ylabel('Threshold Value')
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Histogram with thresholds
    axes[1, 2].hist(anomaly_map_numpy.flatten(), bins=50, alpha=0.7, density=True)
    axes[1, 2].axvline(current_threshold_val, color='red', linestyle='-', linewidth=2,
                       label=f'Selected ({threshold_method})')
    axes[1, 2].axvline(adaptive_thresholds['otsu'], color='blue', linestyle='--', 
                       label='Otsu')
    axes[1, 2].axvline(adaptive_thresholds['statistical'], color='green', linestyle='--', 
                       label='Statistical')
    axes[1, 2].set_title('Anomaly Score Distribution')
    axes[1, 2].set_xlabel('Anomaly Score')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    # Save the comprehensive visualization
    timestamp = int(time.time())
    output_filename = f"anomaly_comparison_{model_name}_{dataset_name}_{timestamp}.png"
    output_path = os.path.join("results", output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive visualization saved to: {output_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n--- Analysis Summary ---")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Threshold method: {threshold_method}")
    print(f"Threshold value: {current_threshold_val:.6f}")
    print(f"Anomalous pixels: {anomaly_percentage:.2f}%")
    print(f"Image dimensions: {input_image.shape}")
    print(f"Anomaly map range: [{anomaly_map_numpy.min():.6f}, {anomaly_map_numpy.max():.6f}]")

if __name__ == "__main__":
    main()
