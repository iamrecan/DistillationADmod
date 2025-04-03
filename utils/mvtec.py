import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MVTecDataset(Dataset):
    def __init__(self, root_dir, resize_shape, crop_size, phase='train'):
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.crop_size = crop_size
        self.phase = phase

        # Get image files
        self.image_files = []
        valid_extensions = {'.png', '.jpg', '.jpeg'}

        if phase == 'train':
            # For training, just look in the root_dir
            for file in os.listdir(root_dir):
                if file.lower().endswith(tuple(valid_extensions)):
                    self.image_files.append(os.path.join(root_dir, file))
        else:
            # For testing, look in all subdirectories
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if file.lower().endswith(tuple(valid_extensions)):
                        self.image_files.append(os.path.join(root, file))

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((resize_shape, resize_shape)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)

            # Determine if image is anomalous based on path
            is_anomaly = 0 if '/good/' in img_path.replace('\\', '/') else 1

            # Return dictionary format as expected by trainer
            return {
                'imageBase': image,
                'has_anomaly': torch.tensor(is_anomaly, dtype=torch.long),
                'path': img_path
            }

        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return {
                'imageBase': torch.zeros(3, self.crop_size, self.crop_size),
                'has_anomaly': torch.tensor(-1, dtype=torch.long),
                'path': img_path
            }