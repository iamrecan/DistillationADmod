import sys
import torch
from PIL import Image
# import sam2  # Assuming sam2 is the correct import for the SAM2 library # Original import
from sam2_for_anomaly_point import SAM2Segmenter # Import local class
import numpy as np
import cv2
import matplotlib.pyplot as plt

np.random.seed(3)  # Moved to the top for proper initialization

# Visualization functions (moved before main)
def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Approximate contours for smoother drawing
        contours = [cv2.approxPolyDP(contour, epsilon=1.0, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image.copy(), contours, -1, (1, 1, 1, 0.5), thickness=1) # Use copy for drawContours
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    coords = np.array(coords)
    labels = np.array(labels)
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        # Convert PIL image to NumPy array for imshow if it's not already,
        # though plt.imshow can often handle PIL Images directly.
        img_display = np.array(image) if isinstance(image, Image.Image) else image
        plt.imshow(img_display)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None, "input_labels must be provided if point_coords are given"
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        # if len(scores) > 1 or scores[i] is not None: # Display score if available # Original condition
        if scores[i] is not None: # Display score if available and not None
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        else:
            plt.title(f"Mask {i+1}", fontsize=18)
        plt.axis('off')
        plt.show()

# Import the SAM2 library (ensure you have installed it in your environment)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
    return device

def configure_device(device):
    if device.type == "cuda":
        # Use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # Turn on tfloat32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

def main():
    if len(sys.argv) != 4:
        print("Usage: python sam2_sementer_from_point.py <image_path> <x> <y>")
        sys.exit(1)

    image_path = sys.argv[1]
    try:
        x = int(sys.argv[2])
        y = int(sys.argv[3])
    except ValueError:
        print("Coordinates must be integers.")
        sys.exit(1)

    # Load the image
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)


    # Select the device for computation
    device = get_device()
    configure_device(device)

    # Initialize the SAM2 segmenter
    try:
        # segmenter = sam2.Sam2Segmenter(model_checkpoint="segment_with_sam2/sam2.1_hiera_small.pt", device=device) # Original instantiation
        # SAM2Segmenter from sam2_for_anomaly_point.py will find "sam2.1_hiera_small.pt" in its directory if model_path is None
        segmenter = SAM2Segmenter(device=device)
    except Exception as e:
        print(f"Error initializing SAM2Segmenter: {e}")
        print("Please ensure the SAM2 library is installed and configured correctly.")
        print("If using a local model, you might need to specify 'model_checkpoint' (e.g., path to .pt file) instead of 'model_type'.")
        sys.exit(1)

    # Prepare points for segmentation and visualization
    image_np = np.array(image) # Convert PIL image to NumPy array for segmenter
    
    # points_for_segmenter = [[x, y]] # Original points format
    points_for_segmenter_np = np.array([[x, y]]) # NumPy array for segmenter
    point_labels_np = np.array([1]) # Label for the point (1 for positive)

    # These are for the show_points function, which expects np.array
    points_for_visualization = np.array([[x, y]])
    labels_for_visualization = np.array([1])  # Assuming the input point is a positive prompt

    print(f"Segmenting image '{image_path}' at point: ({x}, {y})")

    # Perform segmentation using the input point
    try:
        # masks_data = segmenter.segment(image, points=points_for_segmenter) # Original call
        # segment_from_points returns (mask, score)
        best_mask, score = segmenter.segment_from_points(image_np, points=points_for_segmenter_np, point_labels=point_labels_np)
    except Exception as e:
        print(f"Error during segmentation: {e}")
        sys.exit(1)

    # if not masks_data: # Original check
    #     print("No masks were produced by the segmenter.")
    #     return
    if best_mask is None: # New check for single mask
        print("No mask was produced by the segmenter.")
        return

    # For demonstration, print the segmentation masks produced.
    # print("Segmentation masks (raw output):")
    # for idx, mask_item in enumerate(masks_data):
    #     print(f"Mask {idx}: {mask_item}") # mask_item is expected to be a mask array

    # Visualize the masks
    # show_masks expects a list of mask arrays and a list of scores.
    # num_masks = len(masks_data) # Original
    # scores = [1.0] * num_masks # Original dummy scores

    masks_to_show = [best_mask] # List containing the single best mask
    scores_to_show = [score]    # List containing the score for the best mask
    num_masks = 1               # We have one mask

    print(f"Displaying {num_masks} mask(s). Close plot windows to exit.")
    show_masks(image, masks_to_show, scores_to_show,
               point_coords=points_for_visualization,
               input_labels=labels_for_visualization)

if __name__ == '__main__':
    main()

# Removed np.random.seed(3) from here as it was moved to the top
# Removed visualization function definitions from here as they were moved before main