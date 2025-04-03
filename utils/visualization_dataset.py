import torch
import numpy as np
from visualization import Visualizer


def example_usage():
    # Example 1: Plot training curves
    try:
        losses = {
            'total_loss': [0.5, 0.4, 0.3, 0.25, 0.2],
            'reconstruction_loss': [0.3, 0.25, 0.2, 0.15, 0.1],
            'distillation_loss': [0.2, 0.15, 0.1, 0.1, 0.1]
        }
        Visualizer.plot_training_curves(
            losses,
            title="Training Progress",
            save_path="results/training_curves.png"
        )

        # Example 2: Plot anomaly heatmap
        # Create dummy image and anomaly map
        image = torch.randn(3, 224, 224)  # RGB image
        anomaly_map = torch.zeros(224, 224)
        # Add some anomaly regions
        anomaly_map[100:150, 100:150] = 0.8
        anomaly_map[50:70, 160:180] = 0.6

        Visualizer.plot_anomaly_heatmap(
            image,
            anomaly_map,
            save_path="results/anomaly_heatmap.png"
        )

        # Example 3: Plot ROC and PR curves
        # Generate dummy data
        np.random.seed(42)
        n_normal = 1000
        n_anomaly = 100

        # Generate scores
        normal_scores = np.random.normal(0.3, 0.2, n_normal)
        anomaly_scores = np.random.normal(0.7, 0.3, n_anomaly)

        # Combine scores and create labels
        all_scores = np.concatenate([normal_scores, anomaly_scores])
        all_labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])

        Visualizer.plot_roc_pr_curves(
            all_labels,
            all_scores,
            save_path="results/roc_pr_curves.png"
        )

        # Example 4: Plot score distributions
        Visualizer.plot_score_distributions(
            normal_scores,
            anomaly_scores,
            save_path="results/score_distributions.png"
        )
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    example_usage()
