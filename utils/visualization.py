import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import seaborn as sns


class Visualizer:
    @staticmethod
    def plot_training_curves(losses, title="Training Progress", save_path=None):
        """Plot training loss curves.

        Args:
            losses (dict): Dictionary containing different loss components
            title (str): Plot title
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        for name, values in losses.items():
            plt.plot(values, label=name)

        plt.title(title)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_anomaly_heatmap(image, anomaly_map, save_path=None):
        """Plot original image and its anomaly heatmap.

        Args:
            image (torch.Tensor): Original image (C,H,W)
            anomaly_map (torch.Tensor): Anomaly score map (H,W)
            save_path (str, optional): Path to save the plot
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(anomaly_map, torch.Tensor):
            anomaly_map = anomaly_map.cpu().numpy()

        # Convert to displayable format
        if image.shape[0] == 3:  # RGB
            image = np.transpose(image, (1, 2, 0))
        elif image.shape[0] == 1:  # Grayscale
            image = image[0]

        plt.figure(figsize=(12, 4))

        plt.subplot(131)
        plt.title("Original Image")
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(132)
        plt.title("Anomaly Heatmap")
        plt.imshow(anomaly_map, cmap='hot')
        plt.colorbar()
        plt.axis('off')

        plt.subplot(133)
        plt.title("Overlay")
        plt.imshow(image)
        plt.imshow(anomaly_map, cmap='hot', alpha=0.5)
        plt.axis('off')

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_roc_pr_curves(labels, scores, save_path=None):
        """Plot ROC and Precision-Recall curves.

        Args:
            labels (numpy.ndarray): Ground truth labels
            scores (numpy.ndarray): Anomaly scores
            save_path (str, optional): Path to save the plot
        """
        # Calculate ROC curve and ROC area
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        # Calculate Precision-Recall curve and PR area
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(12, 5))

        # Plot ROC curve
        plt.subplot(121)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True)

        # Plot Precision-Recall curve
        plt.subplot(122)
        plt.plot(recall, precision, color='darkorange', lw=2,
                 label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_score_distributions(normal_scores, anomaly_scores, save_path=None):
        """Plot score distributions for normal and anomaly samples.

        Args:
            normal_scores (numpy.ndarray): Anomaly scores for normal samples
            anomaly_scores (numpy.ndarray): Anomaly scores for anomalous samples
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 6))

        sns.kdeplot(data=normal_scores, label='Normal', color='blue')
        sns.kdeplot(data=anomaly_scores, label='Anomaly', color='red')

        plt.title('Score Distributions')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
