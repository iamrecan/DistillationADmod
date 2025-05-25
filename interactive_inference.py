import torch
from PIL import Image
import os
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2 # Added
from pathlib import Path # Added

# Import specific trainers - Add more as needed
from models.SingleNet.trainer_sn import SnTrainer
# from models.DBFAD.trainer_dbfad import DbfadTrainer
# from models.EfficientAD.trainer_ead import EadTrainer
# from models.ReverseDistillation.trainer_rd import RdTrainer
# from models.StudentTeacher.trainer_st import StTrainer


# --- Configuration ---
# These can be changed to use different models and datasets
DATASET_NAME = "wood"  # Or other datasets like "pill", "capsule" etc.
MODEL_TYPE = "sn"      # Or "dbfad", "ead", "rd", "st" based on imported trainers

# Determine MODEL_FILENAME based on MODEL_TYPE, can be expanded
if MODEL_TYPE == "sn":
    MODEL_FILENAME = "student.pth"
else:
    MODEL_FILENAME = "best.pth" # Default for other models

DEFAULT_IMAGE_SIZE = (224, 224) # Should match the training image size for the model
INITIAL_THRESHOLD = 0.5

# Path to the image for inference - user will be prompted later
# IMAGE_PATH = "path/to/your/image.jpg" # This will be prompted
MODEL_RESULTS_BASE_PATH = Path("./results")


# --- New Helper Functions ---

def load_trainer(model_type, config_dict, device):
    """Loads the specified model trainer."""
    print(f"Loading trainer for model type: {model_type}")
    if model_type == "sn":
        return SnTrainer(config_dict, device)
    # Add other model types here as they are imported
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

def visualize_interactive_threshold(original_image_cv2, anomaly_map_numpy, current_threshold, fig_title=""):
    """Visualizes the original image, anomaly map, and thresholded result."""
    plt.figure(figsize=(18, 6))
    if fig_title:
        plt.suptitle(fig_title)

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_image_cv2, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    # Anomaly map
    plt.subplot(1, 3, 2)
    plt.imshow(anomaly_map_numpy, cmap='jet')
    plt.colorbar()
    plt.title("Anomaly Map")
    plt.axis("off")

    # Thresholded result
    plt.subplot(1, 3, 3)
    thresholded_map = (anomaly_map_numpy > current_threshold).astype(np.float32)
    plt.imshow(thresholded_map, cmap='gray')
    plt.title(f"Thresholded Anomaly (>{current_threshold:.3f})")
    plt.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96] if fig_title else None) # Adjust layout if suptitle is used
    plt.show(block=False) # Use non-blocking show for interactive loop
    plt.pause(0.1) # Pause to allow plot to render

# --- Main Script ---
def main():
    print("Starting interactive inference script for threshold selection...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Configure and Load Trainer ---
    # Basic config for the trainer. Some trainers might need more specific params.
    trainer_config = {
        "data_path": str(Path("./dataset")), # Path to the base dataset directory
        "obj": DATASET_NAME,                               # Object/dataset name
        "save_path": str(MODEL_RESULTS_BASE_PATH),         # Base path where models are saved
        "distillType": MODEL_TYPE,                         # Model type
        "model_filename": MODEL_FILENAME,                  # Pass the determined model filename
        "TrainingData": {                                  # TrainingData sub-config
            "img_size": DEFAULT_IMAGE_SIZE[0],
            "crop_size": DEFAULT_IMAGE_SIZE[0], # Assuming crop_size is same as img_size
            "norm": True,
            "epochs": 100, # Added default epochs value
            "lr": 0.001, # Added default learning rate
            "batch_size": 32, # Added default batch size
        }
        # Add any other specific config parameters your trainer/model might need
    }
    
    try:
        trainer = load_trainer(MODEL_TYPE, trainer_config, device)
        # The load_weights method in trainers usually constructs the path internally
        # based on the config (save_path, obj, distillType)
        # For SnTrainer, it specifically loads "student.pth" due to its own implementation.
        # Other trainers might look for "best.pth" or what's passed via "model_filename" if they support it.
        print(f"Attempting to load weights for {MODEL_TYPE} model ({MODEL_FILENAME}) on {DATASET_NAME} dataset...")
        trainer.load_weights() # Assumes trainer has load_weights that uses its internal config or model_filename if adapted
        print("Model weights loaded successfully.")
        trainer.model.eval() # Ensure model is in evaluation mode
    except Exception as e:
        print(f"Error loading trainer or weights: {e}")
        print("Please ensure the MODEL_TYPE, DATASET_NAME are correct, the trainer is implemented in load_trainer,")
        print("and the pre-trained model exists in the expected path:")
        # Use MODEL_FILENAME in the expected path message
        expected_path = MODEL_RESULTS_BASE_PATH / "models" / DATASET_NAME / MODEL_TYPE / MODEL_FILENAME
        print(f"Expected path: {expected_path}")
        return

    # --- 2. Get Image Path from User ---
    while True:
        image_path_str = input("Enter the path to the image for anomaly detection: ")
        image_path = Path(image_path_str)
        if image_path.exists() and image_path.is_file():
            break
        else:
            print(f"Error: Image not found or is not a file at '{image_path_str}'. Please try again.")
    
    print(f"Using image: {image_path}")

    # --- 3. Load and Preprocess Image ---
    input_tensor = preprocess_image_for_trainer(str(image_path), DEFAULT_IMAGE_SIZE)
    if input_tensor is None:
        print("Failed to load or preprocess the image. Exiting.")
        return
    input_tensor = input_tensor.to(device)

    # Load original image with OpenCV for visualization
    try:
        original_image_cv2 = cv2.imread(str(image_path))
        if original_image_cv2 is None:
            raise ValueError("cv2.imread returned None. Check image path and integrity.")
    except Exception as e:
        print(f"Error loading original image with OpenCV from {image_path}: {e}")
        return

    # --- 4. Perform Inference to get Anomaly Map ---
    print("Performing inference to generate anomaly map...")
    try:
        with torch.no_grad():
            trainer.infer(input_tensor) # Call infer method of the trainer
            # The post_process method should return the anomaly map
            # Its shape might vary, common is (1, H, W) or (H, W)
            anomaly_map_raw = trainer.post_process() 
        
        # Process anomaly map (e.g., select first item if batched, move to CPU, convert to numpy)
        if isinstance(anomaly_map_raw, torch.Tensor):
            if anomaly_map_raw.ndim == 4 and anomaly_map_raw.shape[0] == 1 and anomaly_map_raw.shape[1] == 1: # (1,1,H,W)
                anomaly_map_numpy = anomaly_map_raw.squeeze(0).squeeze(0).cpu().numpy()
            elif anomaly_map_raw.ndim == 3 and anomaly_map_raw.shape[0] == 1: # (1,H,W)
                anomaly_map_numpy = anomaly_map_raw.squeeze(0).cpu().numpy()
            elif anomaly_map_raw.ndim == 2: # (H,W)
                anomaly_map_numpy = anomaly_map_raw.cpu().numpy()
            else:
                print(f"Unexpected anomaly map shape: {anomaly_map_raw.shape}. Assuming first element.")
                # This part might need adjustment based on the specific trainer's output
                anomaly_map_numpy = anomaly_map_raw[0].cpu().numpy() if isinstance(anomaly_map_raw, list) else anomaly_map_raw.cpu().numpy()

        elif isinstance(anomaly_map_raw, np.ndarray):
             if anomaly_map_raw.ndim == 3 and anomaly_map_raw.shape[0] == 1: # (1,H,W)
                anomaly_map_numpy = anomaly_map_raw.squeeze(0)
             elif anomaly_map_raw.ndim == 4 and anomaly_map_raw.shape[0] == 1 and anomaly_map_raw.shape[1] == 1: # (1,1,H,W)
                anomaly_map_numpy = anomaly_map_raw.squeeze(0).squeeze(0)
             else: # Assuming (H,W) or needs specific handling
                anomaly_map_numpy = anomaly_map_raw

        else:
            raise TypeError(f"Anomaly map type not recognized: {type(anomaly_map_raw)}")

        print("Inference complete. Anomaly map generated.")
    except Exception as e:
        print(f"Error during model inference or post-processing: {e}")
        print("This could be due to an issue with the model architecture, input tensor shape, or trainer methods.")
        return

    # --- 5. Interactive Thresholding Loop ---
    current_threshold = INITIAL_THRESHOLD
    print("\\n--- Interactive Threshold Adjustment ---")
    while True:
        plt.close() # Close previous plot
        fig_title = f"File: {image_path.name} | Model: {MODEL_TYPE} | Dataset: {DATASET_NAME}"
        visualize_interactive_threshold(original_image_cv2, anomaly_map_numpy, current_threshold, fig_title)
        
        try:
            user_input = input(f"Current threshold: {current_threshold:.4f}. Enter new threshold (e.g., 0.6) or 'q' to quit: ")
            if user_input.lower() == 'q':
                print("Exiting interactive threshold selection.")
                break
            new_threshold = float(user_input)
            current_threshold = new_threshold
        except ValueError:
            print("Invalid input. Please enter a numeric threshold value or 'q'.")
        except Exception as e:
            print(f"An error occurred: {e}")
            break
    
    plt.close() # Close the last plot when done
    print("Script finished.")


if __name__ == "__main__":
    # Remove or comment out old functions if they are no longer needed
    # and were part of the original interactive_inference.py
    # For example, the old load_model, preprocess_image, perform_inference, visualize_results
    # are effectively replaced by the new trainer-based approach and new visualization.
    main()

# --- Old functions (to be removed or commented out if no longer used by main) ---
# def load_model(model_path, device):
#     """Loads the PyTorch model."""
#     print(f"Attempting to load model from: {model_path}")
#     if not os.path.exists(model_path):
#         print(f"Error: Model file not found at {model_path}")
#         return None
#     try:
#         loaded_object = torch.load(model_path, map_location=device)
#         if isinstance(loaded_object, torch.nn.Module):
#             model = loaded_object
#             model.eval()
#             print("Model loaded directly.")
#             return model
#         elif isinstance(loaded_object, dict):
#             print(f"Loaded a dictionary from {model_path}.")
#             print("This script now uses a trainer-based approach. The old load_model is deprecated for this script's main functionality.")
#             return None
#         else:
#             print(f"Unsupported format loaded from {model_path}. Type: {type(loaded_object)}")
#             return None
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return None

# def preprocess_image(image_path, image_size): # Original preprocess_image
#     """Loads and preprocesses an image."""
#     try:
#         image = Image.open(image_path).convert("RGB")
#         transform = T.Compose([
#             T.Resize(image_size),
#             T.ToTensor(),
#             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
#         ])
#         return transform(image).unsqueeze(0) # Add batch dimension
#     except FileNotFoundError:
#         print(f"Error: Image not found at {image_path}")
#         return None
#     except Exception as e:
#         print(f"Error preprocessing image {image_path}: {e}")
#         return None

# def perform_inference(model, input_tensor, device): # Original perform_inference
#     """Performs inference on the input tensor."""
#     if model is None or input_tensor is None:
#         return None
#     input_tensor = input_tensor.to(device)
#     with torch.no_grad():
#         try:
#             output = model(input_tensor)
#             return output
#         except Exception as e:
#             print(f"Error during model inference: {e}")
#             return None

# def visualize_results(image_path, output_tensor): # Original visualize_results
#     """Visualizes the original image and the model's output."""
#     if output_tensor is None:
#         print("Cannot visualize results as model output is None.")
#         return
#     # ... (rest of the old visualize_results function)
#     print("Old visualize_results is deprecated. New visualize_interactive_threshold is used.")
#     pass