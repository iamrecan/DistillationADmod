import torch
import time
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils.util import readYamlConfig
from models.SingleNet.trainer_sn import SnTrainer
from models.DBFAD.trainer_dbfad import DbfadTrainer
from models.EfficientAD.trainer_ead import EadTrainer
from models.ReverseDistillation.trainer_rd import RdTrainer
from models.StudentTeacher.trainer_st import StTrainer
import matplotlib.pyplot as plt
import json


def load_model(model_type, config, device):
    """Load specified model type"""
    if model_type == "sn":
        return SnTrainer(config, device)
    elif model_type == "dbfad":
        return DbfadTrainer(config, device)
    elif model_type == "ead":
        return EadTrainer(config, device)
    elif model_type == "rd":
        return RdTrainer(config, device)
    elif model_type == "st":
        return StTrainer(config, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def measure_inference_time(trainer, sample_image, device, num_runs=100):
    """Measure average inference time"""
    times = []

    # Warmup
    for _ in range(10):
        trainer.infer(sample_image)
        trainer.post_process()

    # Actual timing
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()

            trainer.infer(sample_image)
            trainer.post_process()

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            times.append(end_time - start_time)

    mean_time = np.mean(times)
    std_time = np.std(times)
    return mean_time, std_time


def get_available_models():
    """Get list of available trained models from results directory"""
    results_dir = Path("results/models")
    if not results_dir.exists():
        return []

    available_models = []
    for model_dir in results_dir.iterdir():
        if model_dir.is_dir():
            model_type = model_dir.name
            # Check for checkpoint files
            if list(model_dir.glob("*.pth")):
                available_models.append(model_type)
    return available_models


def get_available_models_by_dataset():
    """Get dictionary of available trained models organized by dataset"""
    results_dir = Path("results/models")
    if not results_dir.exists():
        return {}

    available_models = {}
    # First level directories are datasets (capsule, pill, wood)
    for dataset_dir in results_dir.iterdir():
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name
            available_models[dataset_name] = []

            # Second level contains model types (sn, dbfad, etc)
            for model_dir in dataset_dir.glob("*/*.pth"):
                model_type = model_dir.parent.name
                if model_type not in available_models[dataset_name]:
                    available_models[dataset_name].append(model_type)

    return available_models


def save_and_plot_results(results, save_dir):
    """Save and visualize benchmark results"""
    # Save JSON results
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Extract unique datasets, models and devices
    datasets = set()
    models = set()
    devices = set()
    for key in results.keys():
        dataset, model, device = key.split("_")
        datasets.add(dataset)
        models.add(model)
        devices.add(device)

    # Plot results
    plt.figure(figsize=(15, 8))
    x = np.arange(len(datasets))
    width = 0.15  # Width of bars

    # Plot for each model and device combination
    for i, model in enumerate(models):
        for j, device in enumerate(devices):
            times = []
            errors = []
            for dataset in datasets:
                key = f"{dataset}_{model}_{device}"
                if key in results:
                    times.append(results[key]["mean_time"])
                    errors.append(results[key]["std_time"])
                else:
                    times.append(0)
                    errors.append(0)

            offset = width * (i * len(devices) + j - len(models) * len(devices) / 2)
            plt.bar(x + offset, times, width,
                    label=f"{model.upper()} ({device})",
                    yerr=errors, capsize=5)

    plt.xlabel("Datasets")
    plt.ylabel("Inference Time (ms)")
    plt.title("Model Inference Time Comparison")
    plt.xticks(x, datasets)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "inference_comparison.png", bbox_inches='tight', dpi=300)
    plt.close()


def main():
    # CUDA availability check and info
    print("\nCUDA Information:")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
    print("=" * 50)

    # Get available trained models by dataset
    available_models = get_available_models_by_dataset()
    if not available_models:
        print("No trained models found!")
        return

    print("\nFound trained models by dataset:")
    for dataset, models in available_models.items():
        print(f"{dataset}: {models}")

    # Test on both CPU and CUDA if available
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        # Clear CUDA cache
        torch.cuda.empty_cache()

    # Results dictionary
    results = {}

    for device_name in devices:
        device = torch.device(device_name)
        print(f"\nTesting on {device_name.upper()}")
        print("=" * 50)

        for dataset in available_models.keys():
            print(f"\nDataset: {dataset}")
            print("-" * 30)

            config = {
                "data_path": f"H:\\Tools\\DistillationAD\\datasets\\archive\\{dataset}",
                "obj": dataset,
                "save_path": "./results",
                "distillType": "sn",
                "TrainingData": {
                    "epochs": 100,
                    "batch_size": 32,
                    "lr": 0.0004,
                    "img_size": 224,
                    "crop_size": 224,
                    "norm": True
                }
            }

            # Create sample image
            sample_image = torch.randn(1, 3, 224, 224).to(device)

            for model_type in available_models[dataset]:
                try:
                    print(f"\nTesting {model_type.upper()}")
                    config["distillType"] = model_type

                    # Monitor GPU memory before loading
                    if device.type == "cuda":
                        print(f"GPU memory before loading: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")

                    trainer = load_model(model_type, config, device)
                    model_path = Path(f"results/models/{dataset}/{model_type}/best.pth")

                    if not model_path.exists():
                        print(f"No weights found at {model_path}")
                        continue

                    trainer.load_weights()

                    # Monitor GPU memory after loading
                    if device.type == "cuda":
                        print(f"GPU memory after loading: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")

                    print("Running warmup passes...")
                    mean_time, std_time = measure_inference_time(trainer, sample_image, device)

                    key = f"{dataset}_{model_type}_{device_name}"
                    results[key] = {
                        "mean_time": mean_time * 1000,
                        "std_time": std_time * 1000,
                        "device_info": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
                    }

                    print(f"Average inference time: {mean_time * 1000:.2f} ± {std_time * 1000:.2f} ms")
                    if device.type == "cuda":
                        print(f"Current GPU memory: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")

                    # Clear memory after each model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error testing {model_type}: {str(e)}")
                    continue

    if not results:
        print("\nNo successful model tests completed!")
        return

    # Print detailed summary
    print("\nDetailed Performance Summary")
    print("=" * 80)
    print(f"{'Dataset':10} {'Model':10} {'Device':15} {'Time (ms)':15} {'Device Info':25}")
    print("-" * 80)

    for key, value in results.items():
        dataset, model, device = key.split("_")
        device_info = value.get('device_info', '')
        print(
            f"{dataset:10} {model:10} {device:15} {value['mean_time']:6.2f} ± {value['std_time']:6.2f} {device_info:25}")

    # Save and plot results
    save_dir = "results/inference_benchmarks"
    save_and_plot_results(results, save_dir)
    print(f"\nResults saved to {save_dir}")


if __name__ == "__main__":
    main()