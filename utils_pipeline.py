import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import re
import pickle
import random
import gc
from typing import Optional
from tqdm import tqdm

import tensorflow as tf
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from torch import nn
from torchmetrics import Accuracy, F1Score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, classification_report
from models_pt import get_model




def get_data_loader_pipeline(data_path, dataset_type, batch_size = 128, subset_ratio = 1, device = 'cuda'):

  X_np, y_np, _ = get_data_np_arr(data_path, dataset_type, subset_ratio)

  # Convert your numpy arrays to tensors
  X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)
  y_tensor = torch.tensor(y_np, dtype=torch.long).to(device)

  del X_np, y_np
  gc.collect()

  # Create datasets
  dataset = TensorDataset(X_tensor, y_tensor)

  del X_tensor, y_tensor
  gc.collect()

  # Create DataLoaders
  shuffle_bool = True if dataset_type == 'train' else False
  if device == 'cuda':
    set_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_bool, num_workers=0)
  else:
    set_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_bool, num_workers=0, pin_memory=True)

  del dataset
  gc.collect()

  return set_loader


# converts file date into processed date
def convert_str_file_to_date(filename):
    dt_str1 = filename.split('.')[0][-8:]
    dt_str2 = dt_str1[:4] + '-' + dt_str1[4:6] + '-' + dt_str1[6:]
    return dt_str2


def get_data_np_arr(data_path, set_type, subset_ratio = 1):
  # Initialize empty lists to store the data
  X_list, y_list = [], []
  sequence_numbers_dict = {}

  list_docs = [file_name for file_name in os.listdir(data_path) if (file_name.startswith(set_type) and file_name.endswith('.npz'))]
  total_length = len(list_docs)

  # Calculate the subset size (% of total data)
  subset_length = np.ceil(subset_ratio * total_length).astype(int)

  list_docs = random.sample(list_docs, subset_length)

  # Loop through each .npz file in the folder
  for file_name in sorted(list_docs):

      if file_name.startswith(set_type) and file_name.endswith('.npz'):

          file_path = os.path.join(data_path, file_name)
          test_date = convert_str_file_to_date(file_name)

          # Load the .npz file
          data = np.load(file_path)

          X_list.append(data['X_data'])
          y_list.append(data['y_data'])
          sequence_numbers_dict[test_date] = np.unique(data['sequence_numbers'])

          del data

  # Concatenate all the data from the lists into single numpy arrays
  X = np.concatenate(X_list, axis=0)
  y = np.concatenate(y_list, axis=0)
  all_sequence_numbers_df = pd.DataFrame.from_dict(sequence_numbers_dict, orient='index').transpose()

  del X_list, y_list, sequence_numbers_dict
  import gc
  gc.collect()

  # Display the shapes of the aggregated data
  print(f"X_{set_type} shape: {X.shape}, y_{set_type} shape: {y.shape}, sequence_numbers_{set_type} shape: {all_sequence_numbers_df.shape}")

  return X, y, all_sequence_numbers_df
  
  

def get_subset_dataset_pt(input_dataset, fraction):

  # Get the total length of the dataset
  total_length = len(input_dataset)

  # Calculate the subset size (10% of total data)
  subset_length = int(fraction * total_length)

  # Randomly select 10% of the indices from the dataset
  indices = np.random.choice(range(total_length), subset_length, replace=False)

  # Create a subset of the original dataset
  train_subset = Subset(input_dataset, indices)

  return train_subset



def get_latest_checkpoint_old(model_name, path_csv_training, type_library, seed):
    # Define a regular expression pattern to match checkpoints
    model_name_pattern = re.escape(model_name) + r"_" + f'{type_library}_seed={seed}'
    if type_library == 'tf':
      epoch_pattern = r"-(\d+)"  # This part will capture the epoch number
      val_loss_pattern = r"-(\d+\.\d+)"
      extension_pattern = r"\.keras"
    elif type_library == 'pt':
      epoch_pattern = r"(?:-epoch=(\d+))?"  # Make the epoch part optional
      val_loss_pattern = r"-val_loss=(\d+\.\d+)"  # Captures the validation loss after 'val_loss='
      extension_pattern = r"\.ckpt"  # Matches the '.ckpt' extension

    # Full pattern to match the filenames
    full_pattern = r"^" + model_name_pattern + epoch_pattern + val_loss_pattern + extension_pattern + r"$"

    # List all files in the path
    checkpoint_files = os.listdir(path_csv_training)

    # Filter files that match the checkpoint pattern
    matching_checkpoints = [f for f in checkpoint_files if re.match(full_pattern, f)]

    # If there are no matching checkpoints, return None or raise an error
    if not matching_checkpoints:
        print(f"No checkpoints matched in {checkpoint_files}")
        return None

    # Extract epoch numbers and find the checkpoint with the lowest validation error
    files_with_val_loss = [(f, float(re.search(val_loss_pattern, f).group(1))) for f in matching_checkpoints]

    # Find the file with the minimum validation loss
    lowest_val_loss_file = min(files_with_val_loss, key=lambda x: x[1])[0]
    print(f"The checkpoint with the lowest validation loss is: {lowest_val_loss_file}")

    # Build the full path to the latest checkpoint
    latest_checkpoint_path = os.path.join(path_csv_training, lowest_val_loss_file)

    return latest_checkpoint_path




def get_latest_checkpoint(model_name: str, path_csv_training: str, type_library: str, seed: int) -> Optional[str]:
    """
    Finds the latest checkpoint file with the lowest validation loss.

    Args:
        model_name (str): Name of the model.
        path_csv_training (str): Path to the directory containing checkpoint files.
        type_library (str): Library type ('tf' for TensorFlow, 'pt' for PyTorch).
        seed (int): Seed value used in the model.

    Returns:
        Optional[str]: Full path to the checkpoint file with the lowest validation loss, or None if no matches are found.
    """
    # Construct the base model pattern
    model_name_pattern = re.escape(model_name)

    # Determine patterns based on type_library
    if type_library == 'tf':
        epoch_pattern = r"-(\d+)"  # Captures epoch number
        val_loss_pattern = r"-(\d+\.\d+)"  # Captures validation loss
        extension_pattern = r"\.keras"  # Matches '.keras' extension
    elif type_library == 'pt':
        epoch_pattern = r"_epoch=(\d+)"  # Captures epoch number
        val_loss_pattern = r"_val_loss=(\d+\.\d+)"  # Captures validation loss
        extension_pattern = r"\.ckpt"  # Matches '.ckpt' extension
    else:
        raise ValueError(f"Unsupported library type: {type_library}")

    # Full pattern to match filenames
    full_pattern = rf"^{model_name_pattern}{epoch_pattern}{val_loss_pattern}{extension_pattern}$"

    # List files in the directory
    try:
        checkpoint_files = os.listdir(path_csv_training)
    except FileNotFoundError:
        print(f"Directory not found: {path_csv_training}")
        return None

    # Filter files matching the checkpoint pattern
    matching_checkpoints = [f for f in checkpoint_files if re.match(full_pattern, f)]

    if not matching_checkpoints:
        print(f"No checkpoints matched in directory '{path_csv_training}' with files: {checkpoint_files}")
        return None

    # Extract validation losses and find the checkpoint with the lowest loss
    files_with_val_loss = []
    for f in matching_checkpoints:
        match = re.search(val_loss_pattern, f)
        if match:
            val_loss = float(match.group(1))
            files_with_val_loss.append((f, val_loss))
        else:
            print(f"Warning: Validation loss not found in file: {f}")

    if not files_with_val_loss:
        print(f"No valid validation losses found in matched files: {matching_checkpoints}")
        return None

    # Find the file with the lowest validation loss
    lowest_val_loss_file = min(files_with_val_loss, key=lambda x: x[1])[0]

    # Build the full path to the latest checkpoint
    latest_checkpoint_path = os.path.join(path_csv_training, lowest_val_loss_file)
    print(f"The checkpoint with the lowest validation loss is: {lowest_val_loss_file}")
    
    return latest_checkpoint_path




def get_model_ref_from_ckpt(filename):

    # Regular expression pattern to extract the epoch and val_loss
    pattern = r'epoch=(\d+)_val_loss=([\d\.]+)'
  
    output = None
    
    # Search for the pattern in the filename
    match = re.search(pattern, filename)
    
    if match:
        epoch = match.group(1)
        val_loss = match.group(2)
        output = f'epoch={epoch}-val_loss={val_loss[:-1]}'
    else:
        print("Filename does not match expected syntax")

    return output




def save_checkpoint(epoch, model, optimizer, scheduler, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'model_name': model.name,
        'model_architecture': str(type(model)),
        'optimizer_class': str(type(optimizer)),
        'scheduler_class': str(type(scheduler))
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, device = None, optimizer=None, scheduler=None):
    
    if device == 'cpu':
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path) # Load the checkpoint file
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if optimizer is provided (for training purposes)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state if scheduler is provided (for training purposes)
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', None)
    loss = checkpoint.get('loss', None)
    
    if optimizer is not None and scheduler is not None:
        return model, optimizer, scheduler, epoch, loss
    else:
        return model, epoch, loss


def get_dates_from_folder(doc_name):
  return doc_name.split('_')[2], doc_name.split('_')[3]


def get_trained_model(model_name, checkpoint_path, T=20, device = None, checkpoint_name = None):

  path_csv_training = os.path.dirname(checkpoint_path)

  if checkpoint_name is not None:
    # Path to your saved checkpoint
    #checkpoint_name = 'deeplob_pt_seed=42-epoch=00-val_loss=1.0369.ckpt'   
    checkpoint_path = os.path.join(path_csv_training, checkpoint_name)
    print('not best checkpoint path loaded: ', checkpoint_path)
  
  if os.path.exists(checkpoint_path):

      model = get_model(model_name)
      if model_name == 'binbtabl':
        model = model(120,40,T,5,3,1)
      elif model_name == 'binctabl':
        model = model(120,40,T,5,120,5,3,1)
      else:
        model = model()

      model, last_trained_epoch, last_best_val_loss = load_checkpoint(checkpoint_path, model, device)
      print(f"Loaded model at epoch {last_trained_epoch} with best validation loss: {last_best_val_loss:.4f}")

      return model, last_trained_epoch, last_best_val_loss

  else:
      print(f"Checkpoint file does not exist: {checkpoint_path}")
      return None






def get_pickle_path(model_name, data_path, type_library, seed, device, T):

  path_csv_training = os.path.join(data_path, model_name)

  checkpoint_path = get_latest_checkpoint(model_name, path_csv_training, type_library, seed)

  # get string reference of the model checkpoint for trace in result
  model_ref = get_model_ref_from_ckpt(os.path.basename(checkpoint_path))

  # save test pickle path
  results_pickle_file = f"test_results_{type_library}_seed={seed}_{model_ref}.pkl"
  pickle_test_path = os.path.join(path_csv_training, results_pickle_file)

  loss_fn = nn.CrossEntropyLoss() # cf LOBFrame (Briola et al.)

  model_name = model_name[:-1] if ('bin' in model_name) & (model_name[-1]=='e') else model_name
  model, last_trained_epoch, last_best_val_loss = get_trained_model(model_name, checkpoint_path, T, device)
  dic_model = {'model': model, 'loss_fn':loss_fn, 'last_trained_epoch': last_trained_epoch, 'last_best_val_loss': last_best_val_loss}

  return pickle_test_path, dic_model



# prediction changes vs threshold
def make_prediction(row, threshold):
    # Find the maximum of the three probabilities
    max_prob = max(row['prob_0'], row['prob_1'], row['prob_2'])
    
    # Check which probability is the maximum and apply the threshold logic
    if max_prob == row['prob_0'] and row['prob_0'] > threshold:
        return 0
    elif max_prob == row['prob_2'] and row['prob_2'] > threshold:
        return 2
    else:
        # Default to class 1 if neither prob_0 nor prob_2 exceeds the threshold
        return 1



class Position:
    def __init__(self):
        self.long_inventory = 0
        self.short_inventory = 0
        self.trading_history = []
        self.amount = 1

    def long(self):
        self.long_inventory += self.amount
        self.trading_history.append({'Type': 'Open Long', 'Position':self.long_inventory - self.short_inventory})

    def short(self):
        self.short_inventory += self.amount
        self.trading_history.append({'Type': 'Open Short', 'Position':self.long_inventory - self.short_inventory})

    def exit_long_open_short(self):
        self.long_inventory = 0
        self.short_inventory += self.amount
        self.trading_history.append({r'Type': 'Close Long \ Open Short', 'Position':self.long_inventory - self.short_inventory})

    def exit_short_open_long(self):
        self.short_inventory = 0
        self.long_inventory += self.amount
        self.trading_history.append({r'Type': 'Close Short \ Open Long', 'Position':self.long_inventory - self.short_inventory})
        
    def maintain_long(self):
        self.trading_history.append({'Type': 'Maintain Long', 'Position':self.long_inventory - self.short_inventory})

    def maintain_short(self):
        self.trading_history.append({'Type': 'Maintain Short', 'Position':self.long_inventory - self.short_inventory})

    def no_position(self):
        self.trading_history.append({'Type': 'No Position', 'Position':self.long_inventory - self.short_inventory})


def backtest_positions(labels_arr):

  PositionAgent = Position()

  for i in tqdm(range(len(labels_arr))):

      prediction = labels_arr[i]

      if prediction == 2:
          if PositionAgent.long_inventory == 0 and PositionAgent.short_inventory == 0:
              PositionAgent.long()
          elif PositionAgent.long_inventory == 0 and PositionAgent.short_inventory > 0:
              PositionAgent.exit_short_open_long()
          elif PositionAgent.long_inventory > 0:
              PositionAgent.maintain_long()
      elif prediction == 0:
          if PositionAgent.long_inventory == 0 and PositionAgent.short_inventory == 0:
              PositionAgent.short()
          elif PositionAgent.short_inventory == 0 and PositionAgent.long_inventory > 0:
              PositionAgent.exit_long_open_short()
          elif PositionAgent.short_inventory > 0:
              PositionAgent.maintain_short()
      elif prediction == 1:
          if PositionAgent.long_inventory == 0 and PositionAgent.short_inventory == 0:
              PositionAgent.no_position()
          elif PositionAgent.short_inventory == 0 and PositionAgent.long_inventory > 0:
              PositionAgent.maintain_long()
          elif PositionAgent.short_inventory > 0 and PositionAgent.long_inventory == 0:
              PositionAgent.maintain_short()

  position_history_dataframe = pd.DataFrame(PositionAgent.trading_history)
  return position_history_dataframe



def backtest_position(label, PositionAgent):

  if label == 2:
      if PositionAgent.long_inventory == 0 and PositionAgent.short_inventory == 0:
          PositionAgent.long()
      elif PositionAgent.long_inventory == 0 and PositionAgent.short_inventory > 0:
          PositionAgent.exit_short_open_long()
      elif PositionAgent.long_inventory > 0:
          PositionAgent.maintain_long()
  elif label == 0:
      if PositionAgent.long_inventory == 0 and PositionAgent.short_inventory == 0:
          PositionAgent.short()
      elif PositionAgent.short_inventory == 0 and PositionAgent.long_inventory > 0:
          PositionAgent.exit_long_open_short()
      elif PositionAgent.short_inventory > 0:
          PositionAgent.maintain_short()
  elif label == 1:
      if PositionAgent.long_inventory == 0 and PositionAgent.short_inventory == 0:
          PositionAgent.no_position()
      elif PositionAgent.short_inventory == 0 and PositionAgent.long_inventory > 0:
          PositionAgent.maintain_long()
      elif PositionAgent.short_inventory > 0 and PositionAgent.long_inventory == 0:
          PositionAgent.maintain_short()

  return PositionAgent



def compute_pT(targets_df, predictions):
 
    PositionAgent = Position()

    # Initialize counters
    potential_transactions = 0  # PT
    total_transactions = 0      # TT
    correct_transactions = 0    # CT

    # Initialize previous positions and states
    target_position_curr = targets_df['Position'].iloc[0]
    prediction_position_curr = predictions[0] - 1
    prev_target_position = 0
    prev_prediction_position = 0

    # Loop through each point in the sequence, starting from the second element
    for i in tqdm(range(1, len(predictions))):
        # 
        # Get current positions for targets and predictions
        target_position = targets_df['Position'].iloc[i]
        
        PositionAgent = backtest_position(predictions[i], PositionAgent)
        prediction_position = PositionAgent.trading_history[-1]['Position']

        start_position_match = (prev_target_position == prediction_position_curr)
        current_position_match = (target_position == prediction_position)

        # Detect transition in targets
        if target_position != target_position_curr:
            potential_transactions += 1
            prev_target_position = target_position_curr
            target_position_curr = target_position  # Update start position for next target transition
            
        # Check if a prediction transition matches this target transition
        if prediction_position != prediction_position_curr:
            total_transactions += 1
            prev_prediction_position = prediction_position_curr
            prediction_position_curr = prediction_position  # Update start position for next prediction transition


            # Check if the transitions match in both start and end positions
            if start_position_match and current_position_match:
                correct_transactions += 1


    # Calculate probability of correct transaction pT
    if (potential_transactions + total_transactions - correct_transactions) > 0:
        pT = correct_transactions / (potential_transactions + total_transactions - correct_transactions)
    else:
        pT = 0.0


    return pT





def get_results_stats_by_threshold(results):

  output = {}

  precisions = []
  recalls = []
  f1s = []
  accuracys = []
  mccs = []
  pTs = []
  predictions_list = []

  probabilities_arr = np.array(results['probabilities'])
  df_prob_init = pd.DataFrame()
  for i in range(probabilities_arr.shape[1]):
      df_prob_init[f'prob_{i}'] = probabilities_arr[:,i]

  targets = results['targets']


  for probability_threshold in np.arange(0.3,1,0.1):

      df_prob = df_prob_init.copy(deep = True)
      df_prob['prediction'] = df_prob.apply(make_prediction, axis=1, threshold=probability_threshold)

      probas_arr = df_prob[['prob_0','prob_1','prob_2']].values
      predictions = df_prob['prediction'].values

      position_history_targets = backtest_positions(targets)
      pT = compute_pT(position_history_targets, predictions)

      # statistical evaluations
      report = classification_report(targets, predictions, output_dict=True)
      f1_scores_by_class = np.array(list({class_label: metrics['f1-score'] for class_label, metrics in report.items() if class_label.isdigit()}.values()))
      precision_by_class = np.array(list({class_label: metrics['precision'] for class_label, metrics in report.items() if class_label.isdigit()}.values()))
      recall_by_class = np.array(list({class_label: metrics['recall'] for class_label, metrics in report.items() if class_label.isdigit()}.values()))

      precisions.append(precision_by_class)
      recalls.append(recall_by_class)
      f1s.append(f1_scores_by_class)
      accuracys.append(report['accuracy'])
      mccs.append(round(matthews_corrcoef(targets, predictions), 2))
      pTs.append(pT)
      predictions_list.append(predictions)

  output['precisions'] = precisions
  output['recalls'] = recalls
  output['f1s'] = f1s
  output['accuracys'] = accuracys
  output['mccs'] = mccs
  output['pTs'] = pTs
  output['predictions_list'] = predictions_list

  return output



        


def evaluate_model(dic_model, test_loader, pickle_test_path, train_metrics, val_metrics):

  model = dic_model['model']
  loss_fn = dic_model['loss_fn']
  last_best_val_loss = dic_model['last_best_val_loss']

  print(f'evaluating model: {model.name}, last_best_val_loss = {last_best_val_loss}')

  model.to('cuda')

  # Set the model to evaluation mode
  model.eval()

  # Initialize containers for metrics
  test_loss = 0
  correct_test = 0
  total_test = 0
  all_test_preds = []
  all_test_targets = []
  all_test_probs = []
  batch_loss_test = []

  # Disable gradient calculation for evaluation
  with torch.no_grad():
      for data, target in tqdm(test_loader, desc="Testing", unit="batch"):
          # Move data and target to GPU
          data, target = data.to('cuda'), target.to('cuda')

          # Forward pass
          logits = model(data)
          loss = loss_fn(logits, target)

          # Accumulate loss
          test_loss += loss.item()
          batch_loss_test.append(loss.item())

          # Get probabilities and predictions
          probs = torch.softmax(logits, dim=1)
          preds = torch.argmax(probs, dim=1)

          # Update accuracy and F1 scores
          correct_test += (preds == target).sum().item()
          total_test += target.size(0)

          # Collect all predictions and targets for metrics
          all_test_preds.extend(preds.cpu().tolist())
          all_test_targets.extend(target.cpu().tolist())
          all_test_probs.extend(probs.cpu().tolist())

  # Calculate metrics
  avg_test_loss = np.mean(batch_loss_test)
  test_accuracy = correct_test / total_test

  # You would need to instantiate your metric objects outside this loop
  test_f1_score = val_metrics['f1'](torch.tensor(all_test_preds), torch.tensor(all_test_targets)).item()

  # Log metrics
  print(f"Test Loss: {avg_test_loss:.4f}")
  print(f"Test Accuracy: {test_accuracy:.4f}")
  print(f"Test F1 Score: {test_f1_score:.4f}")

  # Compute confusion matrix
  cm = confusion_matrix(all_test_targets, all_test_preds)
  print(f"Confusion Matrix:\n{cm}")

  # Store results in a dictionary
  results = {
      "outputs": all_test_preds,
      "targets": all_test_targets,
      "probabilities": all_test_probs,
      "confusion_matrix": cm.tolist(),
      "accuracy": test_accuracy,
      "f1_score": test_f1_score,
      'avg_val_loss':last_best_val_loss,
      'avg_test_loss':avg_test_loss
  }
  
  results_stats_dic = get_results_stats_by_threshold(results)
  
  results_stats_dic['f1s_not_weighted'] = [f1_score(results['targets'], pred, average = 'weighted') for pred in         results_stats_dic['predictions_list']]
  results_stats_dic['precisions_not_weighted'] = [precision_score(results['targets'], pred, average = 'weighted') for pred in results_stats_dic['predictions_list']]
  results_stats_dic['recalls_not_weighted'] = [recall_score(results['targets'], pred, average = 'weighted') for pred in results_stats_dic['predictions_list']]
    
  results['stats_by_threshold'] = results_stats_dic
  
  

  # Save results as a pickle file
  with open(pickle_test_path, 'wb') as f:
      pickle.dump(results, f)
      print(f"Test results saved to {pickle_test_path}")

  # Reset the containers (if needed in future runs)
  all_test_preds.clear()
  all_test_targets.clear()
  all_test_probs.clear()
  batch_loss_test.clear()
  
  
  

def evaluate_models(folders_target, data_path, type_library, seed, test_loader, device, T, overwrite = True):

  for model_name in folders_target:

    pickle_test_path, dic_model = get_pickle_path(model_name, data_path, type_library, seed, device, T)

    if os.path.exists(pickle_test_path) and overwrite == False:
      print(f'model {model_name} already evaluated')
      continue

    train_metrics = {'accuracy': Accuracy(task="multiclass", num_classes=3), 'f1': F1Score(task="multiclass", num_classes=3, average="macro")}
    val_metrics = {'accuracy': Accuracy(task="multiclass", num_classes=3), 'f1': F1Score(task="multiclass", num_classes=3, average="macro")}

    evaluate_model(dic_model, test_loader, pickle_test_path, train_metrics, val_metrics)




def evaluate_models_x_symbols_transfer(folders_target, data_path_sym_model, data_path_sym_data, type_library, seed, test_loader, device, T, overwrite = True):

  for model_name in folders_target:

    symbol_model = os.path.basename(os.path.dirname(data_path_sym_model))
    symbol_data = os.path.basename(os.path.dirname(data_path_sym_data))

    pickle_test_path_sym_model, dic_model_trained = get_pickle_path(model_name, data_path_sym_model, type_library, seed, device, T)
    pickle_test_path_sym_data, _ = get_pickle_path(model_name, data_path_sym_data, type_library, seed, device, T)

    trained_model = os.path.basename(pickle_test_path_sym_model)
    trained_model_without_extension = trained_model.split('.pkl')[0]
    save_model = f"{trained_model_without_extension}_trainedOn_{symbol_model}_inferenceOn_{symbol_data}.pkl"
    symbol_data_pickle_path = os.path.dirname(pickle_test_path_sym_data)
    pickle_test_path_sym_data = os.path.join(symbol_data_pickle_path, save_model)

    if os.path.exists(pickle_test_path_sym_data) and overwrite == False:
      print(f'model {model_name} trained on {symbol_model} already evaluated on {symbol_data} data')
      continue

    train_metrics = {'accuracy': Accuracy(task="multiclass", num_classes=3), 'f1': F1Score(task="multiclass", num_classes=3, average="macro")}
    val_metrics = {'accuracy': Accuracy(task="multiclass", num_classes=3), 'f1': F1Score(task="multiclass", num_classes=3, average="macro")}

    evaluate_model(dic_model_trained, test_loader, pickle_test_path_sym_data, train_metrics, val_metrics)



def get_pickled_results(folders_target, data_path, type_library, seed, device, T):

  pickle_test_paths = []
  results_list = []

  for model_name in folders_target:

    pickle_test_path, _ = get_pickle_path(model_name, data_path, type_library, seed, device, T)

    with open(pickle_test_path, 'rb') as f:
      results = pickle.load(f)
    
    pickle_test_paths.append(pickle_test_path)
    results_list.append(results)

  dic_paths_results = {folders_target[i]:{'path_pickle':pickle_test_paths[i], 'results':results_list[i]} for i in range(len(folders_target))}

  return dic_paths_results


def get_eval_metrics(model_name, results):

  accuracy_list = []
  f1_score_move_list = []
  f1_score_stable_list = []
  precision_move_list = []
  precision_stable_list = []
  recall_move_list = []
  recall_stable_list = []
  precision_macro_list = []
  recall_macro_list = []
  f1_score_macro_list = []
  f1_score_weighted_list = []
  precision_weighted_list = []
  recall_weighted_list = []

  cm_list = []
  mcc_list = []

  y_true = np.array(results['targets'])
  probabilities = np.array(results['probabilities'])

  df_proba = pd.DataFrame()
  for i in range(3):
    df_proba[f'prob_{i}'] = probabilities[:,i]


  for p_thres in np.arange(0.3, 1, 0.1).tolist() + [0.95]:

    df_proba['prediction'] = df_proba.apply(make_prediction, axis=1, threshold=p_thres)
    y_pred = df_proba['prediction'].values

    report = classification_report(y_true, y_pred, output_dict=True)

    f1_score_move = 0.5*(report['0']['f1-score'] + report['2']['f1-score'])
    precision_move = 0.5*(report['0']['precision'] + report['2']['precision'])
    recall_move = 0.5*(report['0']['recall'] + report['2']['recall'])
    f1_score_stable = report['1']['f1-score']
    precision_stable = report['1']['precision']
    recall_stable = report['1']['recall']

    precision_macro = report['macro avg']['precision']
    recall_macro = report['macro avg']['recall']
    f1_score_macro = report['macro avg']['f1-score']

    precision_weighted = report['weighted avg']['precision']
    recall_weighted = report['weighted avg']['recall']
    f1_score_weighted = report['weighted avg']['f1-score']

    accuracy = accuracy_score(y_true, y_pred)
    mcc = round(matthews_corrcoef(y_true, y_pred), 2)

    accuracy_list.append(accuracy)
    mcc_list.append(mcc)
    f1_score_move_list.append(f1_score_move)
    f1_score_stable_list.append(f1_score_stable)
    precision_move_list.append(precision_move)
    precision_stable_list.append(precision_stable)
    recall_move_list.append(recall_move)
    recall_stable_list.append(recall_stable)
    precision_macro_list.append(precision_macro)
    recall_macro_list.append(recall_macro)
    f1_score_macro_list.append(f1_score_macro)
    f1_score_weighted_list.append(f1_score_weighted)
    precision_weighted_list.append(precision_weighted)
    recall_weighted_list.append(recall_weighted)

  df_model_metrics = pd.DataFrame({'model':[model_name]*len(accuracy_list), 
                'p_thres':np.arange(0.3, 1, 0.1).tolist() + [0.95],
                'accuracy':accuracy_list,
                'mcc':mcc_list,
                'f1_score_move':f1_score_move_list,
                'f1_score_stable':f1_score_stable_list,
                'precision_move':precision_move_list,
                'precision_stable':precision_stable_list,
                'recall_move':recall_move_list,
                'recall_stable':recall_stable_list,
                'precision_macro':precision_macro_list,
                'recall_macro':recall_macro_list,
                'f1_score_macro':f1_score_macro_list,
                'f1_score_weighted':f1_score_weighted_list,
                'precision_weighted':precision_weighted_list,
                'recall_weighted':recall_weighted_list
                })
                
  df_model_metrics['val_loss'] = results['avg_val_loss']
  df_model_metrics['test_loss'] = results['avg_test_loss']
  
  return df_model_metrics



def compute_improvement(df_model_metrics, reference_model_name):

    reference_model_metrics = df_model_metrics[df_model_metrics['model'] == reference_model_name]

    # Step 2: Compute the improvements for each metric (assuming improvement is relative difference)
    metric_columns = [c for c in df_model_metrics.columns if c not in ['model', 'p_thres']]
    
    # Iterate through all models except the reference model and calculate the improvement
    for metric in metric_columns:
        # Merge reference model metrics with the other models based on p_thres for comparison
        df_model_metrics = df_model_metrics.merge(
            reference_model_metrics[['p_thres', metric]].rename(columns={metric: f"{metric}_reference"}),
            on='p_thres',
            how='left'
        )
        
        # Compute the improvement relative to the reference model
        df_model_metrics[f'improvement_{metric}'] = (
            (df_model_metrics[metric] - df_model_metrics[f"{metric}_reference"]) / df_model_metrics[f"{metric}_reference"]
        ) * 100  # Expressing as a percentage improvement
    
    # Step 3: Drop the intermediate reference columns (optional, if not needed anymore)
    df_model_metrics.drop([f"{metric}_reference" for metric in metric_columns], axis=1, inplace=True)

    return df_model_metrics




def get_evals_metrics(folders_target, dic_paths_results, reference_model_name):
  df_model_metrics_list = []
  for model_name in folders_target:
    df_model_metrics = get_eval_metrics(model_name, dic_paths_results[model_name]['results'])
    df_model_metrics = compute_improvement(df_model_metrics, reference_model_name)
    df_model_metrics_list.append(df_model_metrics)
  df_model_metrics = pd.concat(df_model_metrics_list)
  return df_model_metrics



def plot_results_eval(df_model_metrics, reference_model_name, plot_folder_path):

  figures_path = os.path.join(plot_folder_path, 'figures')
  if not os.path.exists(figures_path):
      os.makedirs(figures_path)

  # Group the metrics by type with their subcategories
  metrics = ['Accuracy','MCC','F1_Score','Precision','Recall','Loss']
  grouped_metrics = {metric:[c for c in df_model_metrics.columns if metric.lower() in c] for metric in metrics}

  # Set the number of columns for subplots (max 2 per row)
  n_cols = 2

  # Plot each group of metrics in subplots
  for group_name, metrics in grouped_metrics.items():
      n_rows = len(metrics) // n_cols + (len(metrics) % n_cols > 0)  # Dynamic number of rows

      # Create a figure with subplots
      fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))  # Adjust height dynamically
      axes = axes.flatten()  # Flatten the axes array for easy access

      # Plot each subcategory in a separate subplot
      for i, metric in enumerate(metrics):
          df_plot = df_model_metrics.copy(deep = True)
          if 'improvement' in metric: 
            df_plot = df_plot.loc[df_plot['model']!= reference_model_name].copy(deep = True)

          sns.lineplot(data=df_plot, x='p_thres', y=metric, hue='model', ax=axes[i])
          axes[i].set_title(f"{metric}")
          axes[i].set_xlabel('p_thres')
          axes[i].set_ylabel(metric)
          axes[i].legend(title='model')
          if 'improvement' in metric: axes[i].axhline(y=0, color='black', linestyle='--')

      # Remove any unused subplots if the number of subplots isn't even
      for j in range(i + 1, len(axes)):
          fig.delaxes(axes[j])

      plt.tight_layout()

      save_path = os.path.join(figures_path, f"{reference_model_name}_{group_name}_metrics.png")
      plt.savefig(save_path)
      #plt.close()  # Close the figure to save memory
      plt.show()





def get_nb_obs(data_path, set_type):

    len_list, y_list = [], []

    list_docs = [file_name for file_name in os.listdir(data_path) if (file_name.startswith(set_type) and file_name.endswith('.npz'))]
   
    # Loop through each .npz file in the folder
    for file_name in sorted(list_docs):

        if file_name.startswith(set_type) and file_name.endswith('.npz'):

            file_path = os.path.join(data_path, file_name)

            # Load the .npz file
            data = np.load(file_path)

            len_list.append(len(data['X_data']))
            del data
            gc.collect()

    # Concatenate all the data from the lists into single numpy arrays
    print(f'{np.sum(len_list)} obs in {set_type} data, mean = {int(np.mean(len_list))}, std = {int(np.std(len_list))}')

    return np.sum(len_list), np.std(len_list)
    


def get_dic_len(data_path, set_type):
    '''
    get the number of observations in each files
    '''
    len_dic = {}
    count = 0

    list_docs = [file_name for file_name in os.listdir(data_path) if (file_name.startswith(set_type) and file_name.endswith('.npz'))]
   
    # Loop through each .npz file in the folder
    for file_name in sorted(list_docs):

        if file_name.startswith(set_type) and file_name.endswith('.npz'):

            file_path = os.path.join(data_path, file_name)

            # Load the .npz file
            data = np.load(file_path)

            len_set = len(data['X_data'])
            len_dic[file_name] = len_set
            count += len_set

            del data
            gc.collect()

    print(f'Number of observations in {set_type} set: {count}')

    return len_dic



def get_dataloader_groups(len_dic, MIN_OBS, MAX_OBS):
  '''
  divide the set of files into groups based on the observation thresholds
  '''

  file_list = list(len_dic.items())
  random.shuffle(file_list)

  # Initialize variables for grouping
  groups = []
  current_group = []
  current_obs_sum = 0

  # Sweep through the files and group them based on the observation thresholds
  for file, obs_count in file_list:
    # Check if adding this file would exceed MAX_OBS
    if current_obs_sum + obs_count > MAX_OBS:
        # If adding this file exceeds MAX_OBS, finalize the current group
        if current_obs_sum >= MIN_OBS:  # Only finalize if the current group meets the MIN_OBS threshold
            groups.append(current_group)  # Append the current group to the list of groups
            current_group = []  # Reset the group
            current_obs_sum = 0  # Reset the observation sum
        
    # Now add the file to the current group
    current_group.append(file)
    current_obs_sum += obs_count

  # If there are remaining files after the loop, add them as the final group
  if current_group:
    groups.append(current_group)

  return groups


def get_np_array_from_group(data_path, group):

  X_list, y_list = [], []

  for file_name in group:

      file_path = os.path.join(data_path, file_name)

      # Load the .npz file
      data = np.load(file_path)

      X_list.append(data['X_data'])
      y_list.append(data['y_data'])

      del data
      gc.collect()

  # Concatenate all the data from the lists into single numpy arrays
  X = np.concatenate(X_list, axis=0)
  y = np.concatenate(y_list, axis=0)

  del X_list, y_list
  gc.collect()

  return X, y


def get_data_loader_from_group(data_path, group, device, batch_size, set_type):

  X_np, y_np = get_np_array_from_group(data_path, group)

  set_loader = get_torch_tensor_T100(X_np, y_np, batch_size, device, set_type)

  del X_np, y_np
  gc.collect()

  return set_loader


def get_torch_tensor_T100(X_np, y_np, batch_size, device, dataset_type):

  # Display the shapes of the aggregated data
  print(f"loader_{dataset_type} on {device} has shape: {X_np.shape}")

  # Convert your numpy arrays to tensors
  X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)
  y_tensor = torch.tensor(y_np, dtype=torch.long).to(device)

  del X_np, y_np
  gc.collect()

  # Create datasets
  dataset = TensorDataset(X_tensor, y_tensor)

  del X_tensor, y_tensor
  gc.collect() if device == 'cpu' else torch.cuda.empty_cache()

  # Create DataLoaders
  shuffle_bool = True if dataset_type == 'train' else False
  if device == 'cuda':
    set_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_bool, num_workers=0)
  else:
    set_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_bool, num_workers=0, pin_memory=True)

  del dataset
  gc.collect() if device == 'cpu' else torch.cuda.empty_cache()

  return set_loader