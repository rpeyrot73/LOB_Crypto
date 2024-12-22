# Core Libraries
import os
import sys
import json
import gc
import pickle
import random
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchmetrics import Accuracy, F1Score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

from utils_pipeline import *



def train(
    model, train_loader, optimizer, loss_fn, device, epoch, num_epochs, scheduler=None
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Returns:
    - Tuple containing average training loss and training accuracy.
    """
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]", unit="batch") as pbar:
        for data, target in pbar:
            data, target = data.to('cuda'), target.to('cuda')

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct_train += (output.argmax(dim=1) == target).sum().item()
            total_train += target.size(0)

            pbar.set_postfix({"train loss": train_loss / total_train})

    if scheduler:
        scheduler.step()

    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = correct_train / total_train

    return avg_train_loss, train_accuracy


def validate(
    model, val_loader, loss_fn, device, epoch=None
) -> Tuple[float, float]:
    """
    Evaluate the model on the validation dataset.

    Returns:
    - Tuple containing average validation loss and validation accuracy.
    """
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        with tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1} [Validation]" if epoch is not None else "Validation",
            unit="batch"
        ) as pbar:
            for data, target in pbar:
                data, target = data.to('cuda'), target.to('cuda')

                output = model(data)
                loss = loss_fn(output, target)
                val_loss += loss.item()
                correct_val += (output.argmax(dim=1) == target).sum().item()
                total_val += target.size(0)

                pbar.set_postfix({"val loss": val_loss / total_val})

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct_val / total_val

    return avg_val_loss, val_accuracy




def train_model(
    model_name: str,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    num_epochs: int,
    device: str,
    checkpoint_dir: str
):
    """
    Train and evaluate the model across all epochs.

    Parameters:
    - model_name: Name of the model.
    - model: The PyTorch model to train.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - optimizer: Optimizer for training.
    - scheduler: Learning rate scheduler.
    - loss_fn: Loss function.
    - num_epochs: Number of epochs to train.
    - device: Device to train on.
    - checkpoint_dir: Directory to save checkpoints.

    Returns:
    - None
    """
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} for model {model_name}")

        # Training
        train_loss, train_accuracy = train(
            model, train_loader, optimizer, loss_fn, device, epoch, num_epochs, scheduler
        )

        # Validation
        val_loss, val_accuracy = validate(
            model, val_loader, loss_fn, device, epoch
        )

        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(
                checkpoint_dir, f"{model_name}_epoch={epoch + 1}_val_loss={val_loss:.4f}.ckpt"
            )
            save_checkpoint(epoch + 1, model, optimizer, scheduler, val_loss, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

    print(f"Completed training for model: {model_name}")




def train_model_large_datasets(
    model_name: str,
    model,
    data_path: str,
    optimizer,
    scheduler,
    loss_fn,
    num_epochs: int,
    device: str,
    checkpoint_dir: str,
    train_metrics: Dict[str, nn.Module],
    val_metrics: Dict[str, nn.Module],
    batch_size: int,
    training_from_scratch: bool = True,
    force_new_training_param: bool = False,
    checkpoint_filename_init: str = "checkpoint",
    path_csv_training: str = "./metrics",
):
    """
    Train and evaluate the model across all epochs for large datasets.

    Parameters:
    - model_name: Name of the model.
    - model: The PyTorch model to train.
    - data_path: Path to the dataset directory.
    - optimizer: Optimizer for training.
    - scheduler: Learning rate scheduler.
    - loss_fn: Loss function.
    - num_epochs: Number of epochs to train.
    - device: Device to train on.
    - checkpoint_dir: Directory to save checkpoints.
    - train_metrics: Metrics for training evaluation.
    - val_metrics: Metrics for validation evaluation.
    - batch_size: Batch size for training and validation.
    - training_from_scratch: Whether to start training from scratch.
    - force_new_training_param: Whether to enforce new training parameters.
    - checkpoint_filename_init: Initial filename for checkpoints.
    - path_csv_training: Path to save training metrics.

    Returns:
    - None
    """
    # Thresholds for file grouping
    MIN_OBS = 500_000
    MAX_OBS = 800_000

    # Get lengths of training and validation data
    len_dic_train = get_dic_len(data_path, "train")
    len_dic_val = get_dic_len(data_path, "val")

    # Group files for training and validation
    groups_files_train = get_dataloader_groups(len_dic_train, MIN_OBS, MAX_OBS)
    groups_files_val = get_dataloader_groups(len_dic_val, MIN_OBS, MAX_OBS)

    # Initialize model
    model = model.to('cuda')
    best_val_loss = float("inf")
    start_epochs = 0

    # Resume training if applicable
    if not training_from_scratch:
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_filename_init}.ckpt")
        if os.path.exists(checkpoint_path):
            model, optimizer, last_scheduler, start_epochs, best_val_loss = load_checkpoint(
                checkpoint_path, model, device, optimizer, scheduler
            )
            print(f"Resuming training from epoch {start_epochs} with best validation loss: {best_val_loss:.4f}")
        if force_new_training_param:
            for param_group in optimizer.param_groups:
                param_group["lr"] = optimizer.defaults["lr"]
            print(f"Learning rate reset to: {optimizer.defaults['lr']}")

    # Training and validation loop
    for epoch in range(start_epochs, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} for model {model_name}")

        # Training phase
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0
        all_train_preds, all_train_targets = [], []

        for group in groups_files_train:
            train_loader = get_data_loader_from_group(data_path, group, device, batch_size, "train")

            group_loss, group_correct, group_total, group_preds, group_targets = 0, 0, 0, [], []

            with tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", unit="batch") as pbar:
                for data, target in pbar:
                    data, target = data.to('cuda'), target.to('cuda')

                    optimizer.zero_grad()
                    output = model(data)
                    loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()

                    # Accumulate metrics
                    group_loss += loss.item()
                    preds = output.argmax(dim=1)
                    group_correct += (preds == target).sum().item()
                    group_total += target.size(0)
                    group_preds.append(preds.cpu())
                    group_targets.append(target.cpu())

                    pbar.set_postfix({"train_loss": group_loss / len(group_preds)})

            epoch_train_loss += group_loss
            correct_train += group_correct
            total_train += group_total
            all_train_preds.append(torch.cat(group_preds))
            all_train_targets.append(torch.cat(group_targets))

            # Free memory
            del train_loader
            gc.collect()

        # Calculate training metrics
        avg_train_loss = epoch_train_loss / len(groups_files_train)
        train_accuracy = correct_train / total_train
        train_f1_score = train_metrics["f1"](
            torch.cat(all_train_preds), torch.cat(all_train_targets)
        ).item()
        print(f"Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1_score:.4f}")

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0
        all_val_preds, all_val_targets = [], []

        for group in groups_files_val:
            val_loader = get_data_loader_from_group(data_path, group, device, batch_size, "val")

            group_loss, group_correct, group_total, group_preds, group_targets = 0, 0, 0, [], []

            with tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}", unit="batch") as pbar:
                with torch.no_grad():
                    for data, target in pbar:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        loss = loss_fn(output, target)

                        group_loss += loss.item()
                        preds = output.argmax(dim=1)
                        group_correct += (preds == target).sum().item()
                        group_total += target.size(0)
                        group_preds.append(preds.cpu())
                        group_targets.append(target.cpu())

                        pbar.set_postfix({"val_loss": group_loss / len(group_preds)})

            epoch_val_loss += group_loss
            correct_val += group_correct
            total_val += group_total
            all_val_preds.append(torch.cat(group_preds))
            all_val_targets.append(torch.cat(group_targets))

            # Free memory
            del val_loader
            gc.collect()

        # Calculate validation metrics
        avg_val_loss = epoch_val_loss / len(groups_files_val)
        val_accuracy = correct_val / total_val
        val_f1_score = val_metrics["f1"](
            torch.cat(all_val_preds), torch.cat(all_val_targets)
        ).item()
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1_score:.4f}")

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(
                checkpoint_dir, f"{model_name}_epoch={epoch + 1}_val_loss={avg_val_loss:.4f}.ckpt"
            )
            save_checkpoint(epoch + 1, model, optimizer, scheduler, avg_val_loss, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        scheduler.step()

    print(f"Completed training for model: {model_name}")






def run_training(config_path: str):
    """
    Main function to execute the training pipeline for multiple models, symbols, T, and prediction horizon values.

    Parameters:
    - config_path: Path to the JSON configuration file.
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    # Hyperparameters and configurations
    batch_size = config.get("batch_size", 128)
    learning_rate = config.get("learning_rate", 2e-4)
    num_epochs = config.get("num_epochs", 10)
    subset_ratio = config.get("subset_ratio", 1.0)
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seed = config.get("seed", 42)
    symbols = config["symbols"]
    models = config["models"]
    type_library = config.get("type_library", "pt")
    data_preprocessed_path = os.path.join(os.getcwd(), config["path_save_dataset"])

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    for symbol in symbols:
        for T, H in list(zip(config["Ts"], config["pred_horizons"])):
            print(f"Starting training pipeline for {symbol} (T={T}, pred_horizon={H})")

            # Find the dataset folder
            folder_name_list = [
                c for c in os.listdir(data_preprocessed_path)
                if (f'_T{T}_H{H}' in c) and (symbol in c)
            ]
            if len(folder_name_list) != 1:
                print(
                    f"Duplicated or missing data folder for T{T}_H{H} and symbol {symbol}: {folder_name_list}"
                )
                continue

            full_path_save = os.path.join(data_preprocessed_path, folder_name_list[0])



            for model_name in models:
                print(f"Training model: {model_name}")

                # Initialize model, optimizer, scheduler, loss function
                model = get_model(model_name)
                if model_name == 'binbtabl':
                    model = model(120,40,T,5,3,1)
                elif model_name == 'binctabl':
                    model = model(120,40,T,5,120,5,3,1)
                else:
                    model = model()
                model = model.to('cuda')

                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                scheduler = StepLR(
                    optimizer,
                    step_size=config.get("step_size", 3),
                    gamma=config.get("gamma", 0.8)
                )
                loss_fn = nn.CrossEntropyLoss()

                # Prepare checkpoint directory
                checkpoint_dir = os.path.join(full_path_save, model_name)
                os.makedirs(checkpoint_dir, exist_ok=True)

                # Train the model

                if (T, H) == (100, 50):  # Replace with your condition for large datasets
                    print(f"Using train_model_large_datasets for {model_name}, T={T}, H={H}")
                    train_model_large_datasets(
                        model_name=model_name,
                        model=model,
                        data_path=full_path_save,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_fn=loss_fn,
                        num_epochs=num_epochs,
                        device=device,
                        checkpoint_dir=checkpoint_dir,
                        train_metrics={
                            "f1": F1Score(task="multiclass", num_classes=3, average="macro")
                        },
                        val_metrics={
                            "f1": F1Score(task="multiclass", num_classes=3, average="macro")
                        },
                        batch_size=batch_size,
                        training_from_scratch=True,  # Change as needed
                        force_new_training_param=False,  # Change as needed
                        checkpoint_filename_init=f"{model_name}_T{T}_H{H}",
                        path_csv_training=checkpoint_dir,
                    )
                else:

                    # Load data
                    print(f"Using train_model for {model_name}, T={T}, H={H}")
                    train_loader = get_data_loader_pipeline(
                        full_path_save, 'train', batch_size, subset_ratio, device
                    )
                    val_loader = get_data_loader_pipeline(
                        full_path_save, 'val', batch_size, subset_ratio, device
                    )

                    train_model(
                        model_name=model_name,
                        model=model,
                        train_loader=train_loader,
                        val_loader=train_loader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_fn=loss_fn,
                        num_epochs=num_epochs,
                        device=device,
                        checkpoint_dir=checkpoint_dir
                    )

                del model, optimizer, scheduler, train_loader, val_loader
                gc.collect()

                print(f"Testing model: {model_name}")
                # Testing section
                test_loader = get_data_loader_pipeline(
                    full_path_save, 'test', batch_size, subset_ratio, device
                )

                folders_target = [
                    f for f in os.listdir(full_path_save) if model_name in f
                ]
                if torch.cuda.is_available():
                    evaluate_models(
                        folders_target, full_path_save, type_library, seed,
                        test_loader, device, T, overwrite=True
                    )

                del test_loader
                gc.collect()


    print("Training pipeline completed.")







def run_pipeline_load_data(config_file):
    """
    Loads data from specified folders, aggregates metrics, and returns the data.

    Parameters:
    - config_file: Path to the JSON configuration file
    - get_latest_checkpoint: Function to retrieve the latest checkpoint path
    - get_model_ref_from_ckpt: Function to extract model reference from the checkpoint name

    Returns:
    - aggregated_data: Dictionary containing aggregated model evaluation metrics
    """

    # Load configuration
    with open(config_file, "r") as f:
        config = json.load(f)

    # Extract parameters from config
    root_path = os.path.join(os.getcwd(), config["path_save_dataset"])
    symbols = config["symbols"]
    models = config["models"]
    type_library = config["type_library"]
    seed = config["seed"]
    plot_output_dir = config["path_save_plots"]

    # Create output directory for plots
    os.makedirs(plot_output_dir, exist_ok=True)

    # Initialize storage
    aggregated_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Traverse each symbol directory
    for symbol in symbols:
        symbol_path = os.path.join(root_path, symbol)

        # Ensure the symbol path exists
        if not os.path.exists(symbol_path):
            print(f"Symbol path does not exist: {symbol_path}")
            continue

        # Traverse prediction horizons and Ts
        for H, T in zip(config["pred_horizons"], config["Ts"]):
            folder_suffix = f'_T{T}_H{H}'

            # Locate result folders
            folders_results = [c for c in os.listdir(root_path) if (symbol in c) and (folder_suffix in c)]
            if len(folders_results) != 1:
                print(f"Duplicated or missing data folder for T{T}_H{H} and symbol {symbol}: {folders_results}")
                break
            else:
                folder_results = folders_results[0]

            time_horizon = f"T{T}_H{H}"
            th = f'H{H}'

            # Process each model
            for model in models:
                model_folder_path = os.path.join(root_path, folder_results, model)
                print(f"Processing: {model_folder_path}")

                if os.path.exists(model_folder_path):
                    # Load checkpoint and test results
                    checkpoint_path = get_latest_checkpoint(model, model_folder_path, type_library, seed)
                    model_ref = get_model_ref_from_ckpt(os.path.basename(checkpoint_path))
                    results_pickle_file = f"test_results_{type_library}_seed={seed}_{model_ref[:-1]}.pkl"
                    pickle_test_path = os.path.join(model_folder_path, results_pickle_file)

                    with open(pickle_test_path, 'rb') as f:
                        results = pickle.load(f)

                    # Calculate additional metrics
                    predictions_list = results['stats_by_threshold']['predictions_list']
                    results['stats_by_threshold']['f1s_not_weighted'] = [f1_score(results['targets'], pred, average='weighted') for pred in predictions_list]
                    results['stats_by_threshold']['precisions_not_weighted'] = [precision_score(results['targets'], pred, average='weighted') for pred in predictions_list]
                    results['stats_by_threshold']['recalls_not_weighted'] = [recall_score(results['targets'], pred, average='weighted') for pred in predictions_list]

                    # Compute Matthews correlation coefficient
                    results['mcc'] = round(matthews_corrcoef(results['targets'], results['outputs']), 2)

                    # Store results
                    aggregated_data[model][th][symbol] = results

    print("Data loading and aggregation completed!")
    return aggregated_data



# Function to add padding to y-axis limits
def add_padding(data, padding=0.05):
    data_min = np.min(data)
    data_max = np.max(data)
    return data_min - padding * abs(data_max - data_min), data_max + padding * abs(data_max - data_min)


def plot_metrics_vs_thresholds(models, aggregated_data, probability_thresholds, metrics,
                               time_horizons, symbols, symbolics, colors, metric_labels, plot_output_dir):
    """
    Plots metrics vs probability thresholds for different models and time horizons.

    Parameters:
    - models: List of model names
    - aggregated_data: Dictionary with aggregated data for each model, time horizon, and symbol
    - probability_thresholds: List or array of thresholds to plot against
    - metrics: List of metrics to evaluate
    - time_horizons: List of time horizons
    - symbols: List of security symbols
    - symbolics: List of matplotlib symbols for different securities
    - colors: List of colors corresponding to the securities
    - metric_labels: Dictionary mapping metric keys to display labels
    - plot_output_dir: Directory to save plots
    """

    for model in models:
        # Initialize plot with a grid of subplots (6x3)
        fig, axes = plt.subplots(len(metrics), len(time_horizons), figsize=(10, 15), sharex=True, sharey=False)
        fig.suptitle(model.upper(), fontsize=12, fontweight='bold', y=0.94)

        # Loop through each metric and time horizon to populate the grid
        for metric_idx, metric in enumerate(metrics):
            for horizon_idx, time_horizon in enumerate(time_horizons):
                ax = axes[metric_idx, horizon_idx]

                # Aggregate data for the specific model, time horizon, and metric
                tabl = np.array([aggregated_data[model][time_horizon][symbol]['stats_by_threshold'][metric]
                                 for symbol in symbols])

                # Compute mean and std deviation across securities
                mean_tabl = tabl.mean(axis=0)
                std_tabl = tabl.std(axis=0)

                # Plot each security's data
                for i, (security, symbol, color) in enumerate(zip(symbols, symbolics, colors)):
                    ax.plot(probability_thresholds, tabl[i], linestyle='None', marker=symbol, color=color,
                            label=security.split('-')[0])

                # Plot the mean with std deviation shading
                ax.plot(probability_thresholds, mean_tabl, color='grey', linewidth=1.5, linestyle='--', label='Mean')
                ax.fill_between(probability_thresholds, mean_tabl - std_tabl, mean_tabl + std_tabl, color='gray', alpha=0.1)

                # Configure labels and titles
                ax.grid(False)
                if metric_idx == len(metrics) - 1 and horizon_idx == 1:
                    ax.set_xlabel('Threshold', fontsize=10, fontweight='bold')
                if horizon_idx == 0:
                    ax.set_ylabel(metric_labels[metric], fontsize=10, fontweight='bold')
                if metric_idx == 0:
                    ax.set_title(time_horizon, fontsize=10, fontweight='bold')

        # Add a global legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles[:5], labels[:5], loc='upper center', ncol=5, fontsize=9, title_fontsize=9,
                   bbox_to_anchor=(0.83, 0.946), handletextpad=0.15, columnspacing=0.6)

        # Save the plot
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.subplots_adjust(wspace=0.23)
        plt.savefig(os.path.join(plot_output_dir, f'metrics_vs_proba_threshold_{model}.png'))
        plt.show()
        #plt.close(fig)  # Close the figure after saving to avoid memory overflow



def plot_class_metrics_vs_thresholds(models, aggregated_data, probability_thresholds, metrics,
                                     time_horizons, securities, security_colors, class_symbols,
                                     metric_labels, plot_output_dir):
    """
    Plots class-wise metrics vs probability thresholds for selected securities.

    Parameters:
    - models: List of model names
    - aggregated_data: Dictionary containing the metrics data
    - probability_thresholds: Array of thresholds
    - metrics: List of metrics to plot
    - time_horizons: List of time horizons
    - securities: List of securities to plot
    - security_colors: Dictionary of colors for securities
    - class_symbols: List of symbols for different classes
    - metric_labels: Dictionary mapping metric keys to display labels
    - plot_output_dir: Directory to save plots
    """

    for model in models:
        # Initialize the figure with subplots
        fig, axes = plt.subplots(len(metrics), len(time_horizons), figsize=(10, 8), sharex=True, sharey=False)
        fig.suptitle(f"{model.upper()}", fontsize=12, fontweight='bold', y=0.95)

        # Populate the subplots
        for metric_idx, metric in enumerate(metrics):
            for horizon_idx, time_horizon in enumerate(time_horizons):
                ax = axes[metric_idx, horizon_idx]

                # Loop over securities and plot class metrics
                for sec_idx, security in enumerate(securities):
                    data = np.array(aggregated_data[model][time_horizon][security]['stats_by_threshold'][metric])

                    for class_idx in range(3):  # Three classes: Down, Stable, Up
                        class_values = data[:, class_idx]
                        ax.plot(
                            probability_thresholds,
                            class_values,
                            linestyle='None',
                            marker=class_symbols[class_idx],
                            color=security_colors[security],
                            markeredgecolor='black',
                            markeredgewidth=0.8,
                            label=f"{security.split('-')[0]} - {['Down', 'Stable', 'Up'][class_idx]}"
                                  if horizon_idx == 0 and metric_idx == 0 else ""
                        )

                # Configure labels and titles
                if metric_idx == len(metrics) - 1 and horizon_idx == 1:
                    ax.set_xlabel('Threshold', fontsize=10, fontweight='bold')
                if horizon_idx == 0:
                    ax.set_ylabel(metric_labels[metric], fontsize=10, fontweight='bold')
                if metric_idx == 0:
                    ax.set_title(time_horizon, fontsize=10, fontweight='bold')

        # Create custom legend
        legend_elements = [
            Line2D([0], [0], marker=class_symbols[0], color=security_colors['BTC-USDT'], label='BTC - Down',
                   markeredgecolor='black', markersize=8),
            Line2D([0], [0], marker=class_symbols[0], color=security_colors['ACM-USDT'], label='ACM - Down',
                   markeredgecolor='black', markersize=8),
            Line2D([0], [0], marker=class_symbols[1], color=security_colors['BTC-USDT'], label='BTC - Stable',
                   markeredgecolor='black', markersize=8),
            Line2D([0], [0], marker=class_symbols[1], color=security_colors['ACM-USDT'], label='ACM - Stable',
                   markeredgecolor='black', markersize=8),
            Line2D([0], [0], marker=class_symbols[2], color=security_colors['BTC-USDT'], label='BTC - Up',
                   markeredgecolor='black', markersize=8),
            Line2D([0], [0], marker=class_symbols[2], color=security_colors['ACM-USDT'], label='ACM - Up',
                   markeredgecolor='black', markersize=8)
        ]

        fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=9, title_fontsize=9,
                   bbox_to_anchor=(0.81, 0.965), columnspacing=0.7)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.subplots_adjust(wspace=0.23)
        plt.savefig(os.path.join(plot_output_dir, f'metrics_byClasses_vs_proba_threshold_{model}_only_BTC_ACM.png'))
        plt.show()
        #plt.close(fig)  # Close to free up memory



def plot_confusion_matrices(models, aggregated_data, symbols, time_horizons, plot_output_dir, figsize=(12, 12)):
    """
    Plots confusion matrices as heatmaps for different models, symbols, and time horizons.

    Parameters:
    - models: List of model names
    - aggregated_data: Dictionary containing confusion matrices
    - symbols: List of security symbols
    - time_horizons: List of prediction time horizons
    - plot_output_dir: Directory to save plots
    - figsize: Tuple defining the figure size
    """

    # Set up colormap for heatmaps
    cmap = sns.color_palette("Blues", as_cmap=True)

    for model in models:
        # Initialize the figure with subplots
        fig, axes = plt.subplots(len(symbols), len(time_horizons), figsize=figsize)
        fig.suptitle(f'{model}', fontsize=16, fontweight='bold', y=0.93)

        # Loop through each symbol and time horizon
        for i, security in enumerate(symbols):
            for j, time_horizon in enumerate(time_horizons):
                # Retrieve and normalize confusion matrix
                confusion_matrix = np.array(
                    aggregated_data[model][time_horizon][security]['confusion_matrix']
                )
                row_sums = confusion_matrix.sum(axis=1, keepdims=True)
                normalized_matrix = confusion_matrix / row_sums

                # Plot the heatmap
                ax = axes[i, j]
                sns.heatmap(
                    normalized_matrix, annot=True, fmt=".2f", cmap=cmap, cbar=False,
                    xticklabels=['Down', 'Stable', 'Up'] if i == len(symbols) - 1 else [],
                    yticklabels=['Down', 'Stable', 'Up'] if j == 0 else [],
                    ax=ax
                )

                # Set titles and labels
                if i == 0:
                    ax.set_title(time_horizon, fontsize=12, fontweight='bold')
                if j == 0:
                    ax.set_ylabel(security.split('-')[0], rotation=0, labelpad=28, fontsize=12, fontweight='bold')
                if i == len(symbols) - 1 and j == 1:
                    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold', labelpad=10)

        # Add "True" label on the left side of the figure
        fig.text(0.0, 0.505, 'True', va='center', ha='center', rotation='vertical', fontsize=12, fontweight='bold')

        # Adjust layout and save the plot
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(plot_output_dir, f'confusion_matrix_{model}.png'))
        plt.show()
        #plt.close(fig)  # Close to free up memory



def run_pipeline_plot(config_file, aggregated_data):
    """
    Loads configuration and runs three plotting functions:
    - plot_metrics_vs_thresholds
    - plot_class_metrics_vs_thresholds
    - plot_confusion_matrices

    Parameters:
    - config_file: Path to the configuration JSON file
    - aggregated_data: Dictionary containing precomputed metrics and confusion matrices
    """

    # Load configuration from JSON
    with open(config_file, "r") as f:
        config = json.load(f)

    # Extract parameters from configuration
    root_path = os.path.join(os.getcwd(), config["path_save_dataset"])
    symbols = config["symbols"]
    models = config["models"]
    type_library = config["type_library"]
    seed = config["seed"]

    probability_thresholds = config['probability_thresholds']
    metrics_threshold = config['metrics']
    time_horizons = [f'H{H}' for H in config['pred_horizons']]

    # Define shared plot parameters
    symbolics = config["plot_symbolics"]
    colors = config["plot_colors"]
    metric_labels_threshold = {
        'f1s_not_weighted': 'F1',
        'mccs': 'MCC',
        'pTs': 'pT',
        'accuracys': 'Accuracy',
        'precisions_not_weighted': 'Precision',
        'recalls_not_weighted': 'Recall'
    }

    # Create directory for saving plots
    plot_output_dir = os.path.join(root_path, 'plots')
    os.makedirs(plot_output_dir, exist_ok=True)

    # Run the first plot function
    plot_metrics_vs_thresholds(
        models, aggregated_data, probability_thresholds, metrics_threshold,
        time_horizons, symbols, symbolics, colors, metric_labels_threshold, plot_output_dir
    )

    # Define parameters for BTC and ACM class imbalance plot
    metrics_class = ['f1s', 'precisions', 'recalls']
    securities = ['BTC-USDT', 'ACM-USDT']
    security_colors = {'BTC-USDT': 'cyan', 'ACM-USDT': 'gold'}
    class_symbols = ['s', 'o', '^']  # Down, Stable, Up
    metric_labels_class = {
        'f1s': 'F1',
        'precisions': 'Precision',
        'recalls': 'Recall'
    }

    # Run the second plot function
    plot_class_metrics_vs_thresholds(
        models, aggregated_data, probability_thresholds, metrics_class,
        time_horizons, securities, security_colors, class_symbols,
        metric_labels_class, plot_output_dir
    )

    # Run the third plot function
    plot_confusion_matrices(models, aggregated_data, symbols, time_horizons, plot_output_dir)

    print("All plots generated successfully!")



def run_training_pipeline(config_file, training = True):

    # train models
    if training:
        run_training(config_file)

    # get results
    aggregated_data = run_pipeline_load_data(config_file)

    # plot results
    run_pipeline_plot(config_file, aggregated_data)



