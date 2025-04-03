import os
import time
import numpy as np
import torch
from tqdm import tqdm
from utils.mvtec import MVTecDataset
from utils.util import AverageMeter, set_seed
from utils.functions import cal_anomaly_maps
from utils.util import computeAUROC
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from pathlib import Path


class BaseTrainer:
    def __init__(self, data, device):

        self.getParams(data, device)
        os.makedirs(self.model_dir, exist_ok=True)
        # You can set seed for reproducibility
        set_seed(42)

        self.load_model()
        self.load_data()
        self.load_optim()

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr * 10,
                                                             epochs=self.num_epochs,
                                                             steps_per_epoch=len(self.train_loader))

    def load_optim(self):
        pass

    def load_model(self):
        pass

    def change_mode(self, period="train"):
        pass

    def load_iter(self):
        pass

    def prepare(self):
        pass

    def infer(self, image, test=False):
        pass

    def computeLoss(self):
        pass

    def save_checkpoint(self):
        pass

    def load_weights(self):
        pass

    def post_process(self):
        pass

    def getParams(self, data, device):
        self.device = device
        self.validation_ratio = 0.2
        self.data_path = data['data_path']
        self.obj = data['obj']
        self.img_resize = data['TrainingData']['img_size']
        self.img_cropsize = data['TrainingData']['crop_size']
        self.num_epochs = data['TrainingData']['epochs']
        self.lr = data['TrainingData']['lr']
        self.batch_size = data['TrainingData']['batch_size']
        self.save_path = data['save_path']
        self.model_dir = f'{self.save_path}/models/{self.obj}'
        self.img_dir = f'{self.save_path}/imgs/{self.obj}'
        self.distillType = data['distillType']
        self.norm = data['TrainingData']['norm']

        # Model specific parameters
        self.modelName = data['backbone'] if 'backbone' in data else None
        self.outIndices = data['out_indice'] if 'out_indice' in data else None
        self.embedDim = data['embedDim'] if 'embedDim' in data else None
        self.lambda1 = data['lambda1'] if 'lambda1' in data else None
        self.lambda2 = data['lambda1'] if 'lambda2' in data else None

    def load_data(self):
        kwargs = ({"num_workers": 8, "pin_memory": True} if torch.cuda.is_available() else {})
        train_dir = os.path.join(self.data_path, "train", "good")

        # Debug information
        print(f"\nDetailed dataset debugging:")
        print(f"{'=' * 50}")
        print(f"Training directory: {train_dir}")

        if not os.path.exists(train_dir):
            raise ValueError(f"Training directory not found: {train_dir}")

        # List actual files
        image_files = [f for f in os.listdir(train_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Number of images found directly: {len(image_files)}")
        print(f"First few images: {image_files[:5] if image_files else 'None'}")

        if not image_files:
            raise ValueError(f"No valid images found in {train_dir}")

        # Create dataset
        train_dataset = MVTecDataset(
            root_dir=train_dir,
            resize_shape=self.img_resize,
            crop_size=self.img_cropsize,
            phase='train'
        )

        if len(train_dataset) == 0:
            raise ValueError("Dataset is empty after MVTecDataset initialization")

        print(f"Dataset size after loading: {len(train_dataset)}")

        # Split dataset
        img_nums = len(train_dataset)
        valid_num = int(img_nums * self.validation_ratio)
        train_num = img_nums - valid_num

        print(f"Splitting dataset - Train: {train_num}, Validation: {valid_num}")

        train_data, val_data = torch.utils.data.random_split(
            train_dataset, [train_num, valid_num]
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            **kwargs
        )

        self.val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=8,
            shuffle=False,
            **kwargs
        )

    def train(self):
        print("training " + self.obj)

        self.change_mode("train")

        best_score = None
        start_time = time.time()
        epoch_time = AverageMeter()
        epoch_bar = tqdm(total=len(self.train_loader) * self.num_epochs, desc="Training", unit="batch")

        for epoch in range(1, self.num_epochs + 1):
            losses = AverageMeter()

            self.load_iter()

            for sample in self.train_loader:
                # Ensure sample is dictionary and extract image
                if not isinstance(sample, dict):
                    raise ValueError("Dataset must return dictionary with 'imageBase' key")

                image = sample['imageBase']
                if not isinstance(image, torch.Tensor):
                    raise ValueError("imageBase must be a tensor")

                image = image.to(self.device)

                self.prepare()

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    self.infer(image)
                    loss = self.computeLoss()
                    losses.update(loss.sum().item(), image.size(0))
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                epoch_bar.set_postfix({"Loss": loss.item(), "Epoch": epoch})
                epoch_bar.update()

            val_loss = self.val(epoch_bar)
            if best_score is None:
                best_score = val_loss
                self.save_checkpoint()
            elif val_loss < best_score:
                best_score = val_loss
                self.save_checkpoint()

            epoch_time.update(time.time() - start_time)
            start_time = time.time()

        epoch_bar.close()
        print("Training completed.")

    def val(self, epoch_bar):
        self.change_mode("eval")
        losses = AverageMeter()

        self.load_iter()

        for sample in self.val_loader:
            image = sample['imageBase'].to(self.device)

            self.prepare()

            with torch.set_grad_enabled(False):
                self.infer(image)
                loss = self.computeLoss()

                losses.update(loss.item(), image.size(0))
        epoch_bar.set_postfix({"Loss": loss.item()})

        return losses.avg

    @torch.no_grad()
    @torch.no_grad()
    def test(self):
        """Test method with visualization of predictions"""
        self.load_weights()
        self.change_mode("eval")

        # Create visualization directory
        vis_dir = Path(self.img_dir) / 'test_results'
        vis_dir.mkdir(parents=True, exist_ok=True)

        kwargs = ({"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {})
        test_dataset = MVTecDataset(
            root_dir=os.path.join(self.data_path, "test"),
            resize_shape=self.img_resize,
            crop_size=self.img_cropsize,
            phase='test'
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

        scores = []
        test_imgs = []
        gt_list = []
        progressBar = tqdm(test_loader, desc="Testing")

        # First pass to compute threshold
        for sample in progressBar:
            label = sample['has_anomaly']
            image = sample['imageBase'].to(self.device)

            self.infer(image)
            self.post_process()
            score = cal_anomaly_maps(self.features_s, self.features_t, self.img_cropsize, self.norm)
            scores.append(score)
            gt_list.extend(label.cpu().numpy())

        # Compute metrics and threshold
        img_roc_auc, threshold = self._compute_and_save_metrics(np.array(scores), np.array(gt_list), vis_dir)

        # Second pass for visualization with threshold
        progressBar = tqdm(enumerate(test_loader), desc="Saving visualizations")
        for idx, sample in progressBar:
            label = sample['has_anomaly']
            image = sample['imageBase'].to(self.device)
            img_path = sample['path'][0]

            score = scores[idx]
            predicted_score = score.reshape(score.shape[0], -1).max()

            self._save_test_visualization(
                image, score, img_path,
                label.item(), predicted_score,
                threshold, vis_dir
            )

        print(f"\nResults saved to: {vis_dir}")
        print(f"ROC-AUC Score: {img_roc_auc:.3f}")
        print(f"Threshold: {threshold:.3f}")

        return img_roc_auc

    def _compute_and_save_metrics(self, scores, gt_list, vis_dir):
        """Compute metrics and save ROC curve"""
        # Calculate ROC-AUC using existing function
        img_roc_auc, img_scores = computeAUROC(scores, gt_list, self.obj, " " + self.distillType)

        # Calculate threshold using Otsu's method
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(img_scores)

        # Calculate ROC curve data manually
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(gt_list, img_scores)

        # plot roc curve here
        plt.figure(figsize=(10, 10))
        # Plot ROC curve
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {img_roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(vis_dir / 'roc_curve.png')
        plt.close()

        print(f"\nResults saved to: {vis_dir}")
        print(f"ROC-AUC Score: {img_roc_auc:.3f}")

        return img_roc_auc, threshold

    def _save_test_visualization(self, image, score, img_path, gt_label, predicted_score, threshold, vis_dir):
        """
        Save visualization of test results with predictions

        Args:
            image: Input image tensor
            score: Anomaly score map
            img_path: Path to original image
            gt_label: Ground truth label (0: normal, 1: anomaly)
            predicted_score: Model's predicted anomaly score
            threshold: Otsu threshold for classification
            vis_dir: Directory to save visualizations
        """
        # Determine ground truth and prediction
        is_anomaly_gt = gt_label == 1
        is_anomaly_pred = predicted_score > threshold

        # Set visualization status
        gt_status = 'NOK' if is_anomaly_gt else 'OK'
        pred_status = 'NOK' if is_anomaly_pred else 'OK'
        status_color = 'red' if is_anomaly_pred else 'green'

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'GT: {gt_status} | Pred: {pred_status} | Score: {predicted_score:.6f}',
                     color=status_color, fontsize=16, y=1.05)

        # Original image
        img_np = vutils.make_grid(image, normalize=True).cpu().numpy().transpose(1, 2, 0)
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Anomaly map
        anomaly_map = score.reshape(self.img_cropsize, self.img_cropsize)
        im = axes[1].imshow(anomaly_map, cmap='jet')
        axes[1].set_title(f'Anomaly Map (Score: {predicted_score:.3f})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])

        # Overlay
        axes[2].imshow(img_np)
        axes[2].imshow(anomaly_map, cmap='jet', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        # Add colored borders
        for ax in axes:
            for spine in ax.spines.values():
                spine.set_edgecolor(status_color)
                spine.set_linewidth(2)

        # Save figure with detailed filename
        img_name = Path(img_path).stem
        match_status = 'Match' if is_anomaly_gt == is_anomaly_pred else 'Mismatch'
        save_path = vis_dir / f'{img_name}_GT-{gt_status}_Pred-{pred_status}_{match_status}.png'
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()