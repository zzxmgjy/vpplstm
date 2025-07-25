#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automatic Hyperparameter Tuning for Power Prediction Model
No external dependencies beyond what's already in train_encdec_7d_single_station.py
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
import joblib
import itertools
import random
from sklearn.metrics import mean_absolute_percentage_error
import torch
import warnings
import copy
import sys

# Import from the original training script
from train_encdec_7d_single_station import main as train_model
from train_encdec_7d_single_station import CFG

warnings.filterwarnings("ignore")

def grid_search(param_grid, station_id=None, data_file='merged_station_test.csv', n_trials=20, random_search=True):
    """
    Perform grid search or random search over the parameter grid.
    
    Args:
        param_grid: Dictionary of parameter names and possible values
        station_id: Station ID to train model for
        data_file: Input data file
        n_trials: Maximum number of trials to run
        random_search: If True, randomly sample from param_grid; if False, try all combinations
    
    Returns:
        best_params: Dictionary of best parameters
        best_mape: Best MAPE value achieved
    """
    print(f"Starting hyperparameter tuning for station: {station_id}")
    print(f"Using data file: {data_file}")
    print(f"Maximum number of trials: {n_trials}")
    
    # Create study directory
    study_dir = f"hyperparameter_tuning_results"
    os.makedirs(study_dir, exist_ok=True)
    
    # Generate parameter combinations
    if random_search:
        # Random search - randomly sample from parameter grid
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generate random combinations
        param_combinations = []
        for _ in range(min(n_trials, 100)):  # Cap at 100 trials max
            combination = {}
            for key, values in param_grid.items():
                combination[key] = random.choice(values)
            param_combinations.append(combination)
    else:
        # Grid search - try all combinations
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = [dict(zip(param_keys, combo)) for combo in itertools.product(*param_values)]
        
        # If too many combinations, randomly sample
        if len(param_combinations) > n_trials:
            random.shuffle(param_combinations)
            param_combinations = param_combinations[:n_trials]
    
    print(f"Generated {len(param_combinations)} parameter combinations to try")
    
    # Track results
    results = []
    best_mape = float('inf')
    best_params = None
    
    # Run trials
    for trial_num, params in enumerate(param_combinations, 1):
        print(f"\nTrial {trial_num}/{len(param_combinations)}:")
        print(f"Parameters: {params}")
        
        # Update global CFG
        cfg_backup = copy.deepcopy(CFG)
        for key, value in params.items():
            if key in CFG:
                CFG[key] = value
        
        # Train model with current parameters
        try:
            output_dir = train_model(station_id, data_file)
            
            # Load MAPE results
            mape_results = joblib.load(f'{output_dir}/mape_results.pkl')
            overall_power_mape = mape_results['overall_power_mape']
            
            # Track result
            result = {
                'trial': trial_num,
                'params': params,
                'mape': overall_power_mape,
                'output_dir': output_dir
            }
            results.append(result)
            
            print(f"Trial {trial_num} completed with MAPE: {overall_power_mape:.2%}")
            
            # Update best parameters if better
            if overall_power_mape < best_mape:
                best_mape = overall_power_mape
                best_params = params
                print(f"New best MAPE: {best_mape:.2%}")
        except Exception as e:
            print(f"Trial {trial_num} failed: {str(e)}")
            # Restore CFG
            for key, value in cfg_backup.items():
                CFG[key] = value
            continue
        
        # Restore CFG
        for key, value in cfg_backup.items():
            CFG[key] = value
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{study_dir}/tuning_results_station_{station_id}.csv", index=False)
    
    # Save best parameters
    if best_params:
        best_params_with_mape = best_params.copy()
        best_params_with_mape['best_mape'] = best_mape
        joblib.dump(best_params_with_mape, f"{study_dir}/best_params_station_{station_id}.pkl")
    
    # Print final results
    print("\n" + "="*50)
    print("Hyperparameter Tuning Results")
    print("="*50)
    if best_params:
        print(f"Best MAPE: {best_mape:.2%}")
        print("\nBest hyperparameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
    else:
        print("No successful trials completed.")
    
    return best_params, best_mape

def apply_best_params(station_id=None, data_file='merged_station_test.csv', params=None):
    """
    Apply best parameters and train a model.
    
    Args:
        station_id: Station ID to train model for
        data_file: Input data file
        params: Dictionary of parameters to apply
    
    Returns:
        output_dir: Directory where the model is saved
    """
    if not params:
        # Default optimized parameters
        params = {
            'hidden_dim': 384,
            'num_layers': 2,
            'drop_rate': 0.3,
            'batch_size': 128,
            'lr': 5e-4,
            'top_k': 90,
            'use_stl': True,
            'power_weight': 1.2,
            'not_use_power_weight': 0.7,
        }
    
    # Print parameters
    print("\nApplying parameters:")
    for key, value in params.items():
        if key != 'best_mape':
            print(f"  {key}: {value}")
            if key in CFG:
                CFG[key] = value
    
    # Train model with best parameters
    print("\nTraining model with best parameters...")
    output_dir = train_model(station_id, data_file)
    print(f"Model saved to: {output_dir}")
    
    return output_dir

def incremental_finetune(model_dir, station_id=None, data_file='merged_station_test.csv', 
                         epochs=50, lr=1e-5, focus_days=None):
    """
    Incrementally fine-tune a pre-trained model with a focus on specific days.
    
    Args:
        model_dir: Directory containing the pre-trained model
        station_id: Station ID to fine-tune for
        data_file: Input data file
        epochs: Number of fine-tuning epochs
        lr: Learning rate for fine-tuning
        focus_days: List of days to focus on (1-7), or None for all days
    
    Returns:
        output_dir: Directory where the fine-tuned model is saved
        overall_power_mape: Overall MAPE for total_active_power
    """
    print(f"Loading model from {model_dir}...")
    
    # Load model components
    sc_e = joblib.load(f'{model_dir}/scaler_enc.pkl')
    sc_d = joblib.load(f'{model_dir}/scaler_dec.pkl')
    sc_y_power = joblib.load(f'{model_dir}/scaler_y_power.pkl')
    sc_y_not_use_power = joblib.load(f'{model_dir}/scaler_y_not_use_power.pkl')
    ENC_COLS = joblib.load(f'{model_dir}/enc_cols.pkl')
    DEC_COLS = joblib.load(f'{model_dir}/dec_cols.pkl')
    config = joblib.load(f'{model_dir}/config.pkl')
    
    # Define the model architecture to match exactly what's in train_encdec_7d_single_station.py
    class EncDecPowerModel(torch.nn.Module):
        def __init__(self, d_enc, d_dec, hid, drop, num_layers=2):
            super().__init__()
            self.enc = torch.nn.LSTM(d_enc, hid, num_layers=num_layers, batch_first=True, 
                              dropout=drop if num_layers>1 else 0)
            self.dec = torch.nn.LSTM(d_dec, hid, num_layers=num_layers, batch_first=True, 
                              dropout=drop if num_layers>1 else 0)
            self.dp = torch.nn.Dropout(drop)
            
            # Enhanced middle layers
            self.fc_mid = torch.nn.Sequential(
                torch.nn.Linear(hid, hid//2),
                torch.nn.ReLU(),
                torch.nn.Dropout(drop),
                torch.nn.Linear(hid//2, hid//4),
                torch.nn.ReLU(),
                torch.nn.Dropout(drop)
            )
            
            self.fc_power = torch.nn.Linear(hid//4, 1)           # total_active_power prediction
            self.fc_not_use_power = torch.nn.Linear(hid//4, 1)   # not_use_power prediction
            
        def forward(self, xe, xd):
            _, (h, c) = self.enc(xe)
            out, _ = self.dec(xd, (h, c))
            out_dp = self.dp(out)
            out_mid = self.fc_mid(out_dp)
            power_pred = self.fc_power(out_mid).squeeze(-1)
            not_use_power_pred = self.fc_not_use_power(out_mid).squeeze(-1)
            return power_pred, not_use_power_pred
    
    # Create model with the same architecture
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncDecPowerModel(
        len(ENC_COLS), 
        len(DEC_COLS), 
        config['hidden_dim'], 
        config['drop_rate'], 
        config['num_layers']
    ).to(dev)
    
    # Load model weights
    model.load_state_dict(torch.load(f'{model_dir}/model_power.pth', map_location=dev))
    
    # Create output directory
    if focus_days:
        focus_days_str = '_'.join(map(str, focus_days))
        output_dir = f"{model_dir}_finetuned_days_{focus_days_str}"
    else:
        output_dir = f"{model_dir}_finetuned"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the incremental_finetune script directly
    from incremental_finetune import incremental_finetune as inc_finetune
    
    return inc_finetune(model_dir, station_id, data_file, epochs, lr, focus_days)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automatic Hyperparameter Tuning for Power Prediction Model
No external dependencies beyond what's already in train_encdec_7d_single_station.py
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
import joblib
import itertools
import random
from sklearn.metrics import mean_absolute_percentage_error
import torch
import warnings
import copy
import sys

# Import from the original training script
from train_encdec_7d_single_station import main as train_model
from train_encdec_7d_single_station import CFG

warnings.filterwarnings("ignore")

def grid_search(param_grid, station_id=None, data_file='merged_station_test.csv', n_trials=20, random_search=True):
    """
    Perform grid search or random search over the parameter grid.
    
    Args:
        param_grid: Dictionary of parameter names and possible values
        station_id: Station ID to train model for
        data_file: Input data file
        n_trials: Maximum number of trials to run
        random_search: If True, randomly sample from param_grid; if False, try all combinations
    
    Returns:
        best_params: Dictionary of best parameters
        best_mape: Best MAPE value achieved
    """
    print(f"Starting hyperparameter tuning for station: {station_id}")
    print(f"Using data file: {data_file}")
    print(f"Maximum number of trials: {n_trials}")
    
    # Create study directory
    study_dir = f"hyperparameter_tuning_results"
    os.makedirs(study_dir, exist_ok=True)
    
    # Generate parameter combinations
    if random_search:
        # Random search - randomly sample from parameter grid
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generate random combinations
        param_combinations = []
        for _ in range(min(n_trials, 100)):  # Cap at 100 trials max
            combination = {}
            for key, values in param_grid.items():
                combination[key] = random.choice(values)
            param_combinations.append(combination)
    else:
        # Grid search - try all combinations
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = [dict(zip(param_keys, combo)) for combo in itertools.product(*param_values)]
        
        # If too many combinations, randomly sample
        if len(param_combinations) > n_trials:
            random.shuffle(param_combinations)
            param_combinations = param_combinations[:n_trials]
    
    print(f"Generated {len(param_combinations)} parameter combinations to try")
    
    # Track results
    results = []
    best_mape = float('inf')
    best_params = None
    
    # Run trials
    for trial_num, params in enumerate(param_combinations, 1):
        print(f"\nTrial {trial_num}/{len(param_combinations)}:")
        print(f"Parameters: {params}")
        
        # Update global CFG
        cfg_backup = copy.deepcopy(CFG)
        for key, value in params.items():
            if key in CFG:
                CFG[key] = value
        
        # Train model with current parameters
        try:
            output_dir = train_model(station_id, data_file)
            
            # Load MAPE results
            mape_results = joblib.load(f'{output_dir}/mape_results.pkl')
            overall_power_mape = mape_results['overall_power_mape']
            
            # Track result
            result = {
                'trial': trial_num,
                'params': params,
                'mape': overall_power_mape,
                'output_dir': output_dir
            }
            results.append(result)
            
            print(f"Trial {trial_num} completed with MAPE: {overall_power_mape:.2%}")
            
            # Update best parameters if better
            if overall_power_mape < best_mape:
                best_mape = overall_power_mape
                best_params = params
                print(f"New best MAPE: {best_mape:.2%}")
        except Exception as e:
            print(f"Trial {trial_num} failed: {str(e)}")
            # Restore CFG
            for key, value in cfg_backup.items():
                CFG[key] = value
            continue
        
        # Restore CFG
        for key, value in cfg_backup.items():
            CFG[key] = value
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{study_dir}/tuning_results_station_{station_id}.csv", index=False)
    
    # Save best parameters
    if best_params:
        best_params_with_mape = best_params.copy()
        best_params_with_mape['best_mape'] = best_mape
        joblib.dump(best_params_with_mape, f"{study_dir}/best_params_station_{station_id}.pkl")
    
    # Print final results
    print("\n" + "="*50)
    print("Hyperparameter Tuning Results")
    print("="*50)
    if best_params:
        print(f"Best MAPE: {best_mape:.2%}")
        print("\nBest hyperparameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
    else:
        print("No successful trials completed.")
    
    return best_params, best_mape

def apply_best_params(station_id=None, data_file='merged_station_test.csv', params=None):
    """
    Apply best parameters and train a model.
    
    Args:
        station_id: Station ID to train model for
        data_file: Input data file
        params: Dictionary of parameters to apply
    
    Returns:
        output_dir: Directory where the model is saved
    """
    if not params:
        # Default optimized parameters
        params = {
            'hidden_dim': 384,
            'num_layers': 2,
            'drop_rate': 0.3,
            'batch_size': 128,
            'lr': 5e-4,
            'top_k': 90,
            'use_stl': True,
            'power_weight': 1.2,
            'not_use_power_weight': 0.7,
        }
    
    # Print parameters
    print("\nApplying parameters:")
    for key, value in params.items():
        if key != 'best_mape':
            print(f"  {key}: {value}")
            if key in CFG:
                CFG[key] = value
    
    # Train model with best parameters
    print("\nTraining model with best parameters...")
    output_dir = train_model(station_id, data_file)
    print(f"Model saved to: {output_dir}")
    
    return output_dir

def incremental_finetune(model_dir, station_id=None, data_file='merged_station_test.csv', 
                         epochs=50, lr=1e-5, focus_days=None):
    """
    Incrementally fine-tune a pre-trained model with a focus on specific days.
    
    Args:
        model_dir: Directory containing the pre-trained model
        station_id: Station ID to fine-tune for
        data_file: Input data file
        epochs: Number of fine-tuning epochs
        lr: Learning rate for fine-tuning
        focus_days: List of days to focus on (1-7), or None for all days
    
    Returns:
        output_dir: Directory where the fine-tuned model is saved
        overall_power_mape: Overall MAPE for total_active_power
    """
    print(f"Loading model from {model_dir}...")
    
    # Load model components
    sc_e = joblib.load(f'{model_dir}/scaler_enc.pkl')
    sc_d = joblib.load(f'{model_dir}/scaler_dec.pkl')
    sc_y_power = joblib.load(f'{model_dir}/scaler_y_power.pkl')
    sc_y_not_use_power = joblib.load(f'{model_dir}/scaler_y_not_use_power.pkl')
    ENC_COLS = joblib.load(f'{model_dir}/enc_cols.pkl')
    DEC_COLS = joblib.load(f'{model_dir}/dec_cols.pkl')
    config = joblib.load(f'{model_dir}/config.pkl')
    
    # Define the model architecture to match exactly what's in train_encdec_7d_single_station.py
    class EncDecPowerModel(torch.nn.Module):
        def __init__(self, d_enc, d_dec, hid, drop, num_layers=2):
            super().__init__()
            self.enc = torch.nn.LSTM(d_enc, hid, num_layers=num_layers, batch_first=True, 
                              dropout=drop if num_layers>1 else 0)
            self.dec = torch.nn.LSTM(d_dec, hid, num_layers=num_layers, batch_first=True, 
                              dropout=drop if num_layers>1 else 0)
            self.dp = torch.nn.Dropout(drop)
            
            # Enhanced middle layers
            self.fc_mid = torch.nn.Sequential(
                torch.nn.Linear(hid, hid//2),
                torch.nn.ReLU(),
                torch.nn.Dropout(drop),
                torch.nn.Linear(hid//2, hid//4),
                torch.nn.ReLU(),
                torch.nn.Dropout(drop)
            )
            
            self.fc_power = torch.nn.Linear(hid//4, 1)           # total_active_power prediction
            self.fc_not_use_power = torch.nn.Linear(hid//4, 1)   # not_use_power prediction
            
        def forward(self, xe, xd):
            _, (h, c) = self.enc(xe)
            out, _ = self.dec(xd, (h, c))
            out_dp = self.dp(out)
            out_mid = self.fc_mid(out_dp)
            power_pred = self.fc_power(out_mid).squeeze(-1)
            not_use_power_pred = self.fc_not_use_power(out_mid).squeeze(-1)
            return power_pred, not_use_power_pred
    
    # Create model with the same architecture
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncDecPowerModel(
        len(ENC_COLS), 
        len(DEC_COLS), 
        config['hidden_dim'], 
        config['drop_rate'], 
        config['num_layers']
    ).to(dev)
    
    # Load model weights
    model.load_state_dict(torch.load(f'{model_dir}/model_power.pth', map_location=dev))
    
    # Create output directory
    if focus_days:
        focus_days_str = '_'.join(map(str, focus_days))
        output_dir = f"{model_dir}_finetuned_days_{focus_days_str}"
    else:
        output_dir = f"{model_dir}_finetuned"
    
    os.makedirs(output_dir, exist_ok=True)
    
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automatic Hyperparameter Tuning for Power Prediction Model
No external dependencies beyond what's already in train_encdec_7d_single_station.py
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
import joblib
import itertools
import random
from sklearn.metrics import mean_absolute_percentage_error
import torch
import warnings
import copy
import sys

# Import from the original training script
from train_encdec_7d_single_station import main as train_model
from train_encdec_7d_single_station import CFG

warnings.filterwarnings("ignore")

def grid_search(param_grid, station_id=None, data_file='merged_station_test.csv', n_trials=20, random_search=True):
    """
    Perform grid search or random search over the parameter grid.
    
    Args:
        param_grid: Dictionary of parameter names and possible values
        station_id: Station ID to train model for
        data_file: Input data file
        n_trials: Maximum number of trials to run
        random_search: If True, randomly sample from param_grid; if False, try all combinations
    
    Returns:
        best_params: Dictionary of best parameters
        best_mape: Best MAPE value achieved
    """
    print(f"Starting hyperparameter tuning for station: {station_id}")
    print(f"Using data file: {data_file}")
    print(f"Maximum number of trials: {n_trials}")
    
    # Create study directory
    study_dir = f"hyperparameter_tuning_results"
    os.makedirs(study_dir, exist_ok=True)
    
    # Generate parameter combinations
    if random_search:
        # Random search - randomly sample from parameter grid
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generate random combinations
        param_combinations = []
        for _ in range(min(n_trials, 100)):  # Cap at 100 trials max
            combination = {}
            for key, values in param_grid.items():
                combination[key] = random.choice(values)
            param_combinations.append(combination)
    else:
        # Grid search - try all combinations
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = [dict(zip(param_keys, combo)) for combo in itertools.product(*param_values)]
        
        # If too many combinations, randomly sample
        if len(param_combinations) > n_trials:
            random.shuffle(param_combinations)
            param_combinations = param_combinations[:n_trials]
    
    print(f"Generated {len(param_combinations)} parameter combinations to try")
    
    # Track results
    results = []
    best_mape = float('inf')
    best_params = None
    
    # Run trials
    for trial_num, params in enumerate(param_combinations, 1):
        print(f"\nTrial {trial_num}/{len(param_combinations)}:")
        print(f"Parameters: {params}")
        
        # Update global CFG
        cfg_backup = copy.deepcopy(CFG)
        for key, value in params.items():
            if key in CFG:
                CFG[key] = value
        
        # Train model with current parameters
        try:
            output_dir = train_model(station_id, data_file)
            
            # Load MAPE results
            mape_results = joblib.load(f'{output_dir}/mape_results.pkl')
            overall_power_mape = mape_results['overall_power_mape']
            
            # Track result
            result = {
                'trial': trial_num,
                'params': params,
                'mape': overall_power_mape,
                'output_dir': output_dir
            }
            results.append(result)
            
            print(f"Trial {trial_num} completed with MAPE: {overall_power_mape:.2%}")
            
            # Update best parameters if better
            if overall_power_mape < best_mape:
                best_mape = overall_power_mape
                best_params = params
                print(f"New best MAPE: {best_mape:.2%}")
        except Exception as e:
            print(f"Trial {trial_num} failed: {str(e)}")
            # Restore CFG
            for key, value in cfg_backup.items():
                CFG[key] = value
            continue
        
        # Restore CFG
        for key, value in cfg_backup.items():
            CFG[key] = value
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{study_dir}/tuning_results_station_{station_id}.csv", index=False)
    
    # Save best parameters
    if best_params:
        best_params_with_mape = best_params.copy()
        best_params_with_mape['best_mape'] = best_mape
        joblib.dump(best_params_with_mape, f"{study_dir}/best_params_station_{station_id}.pkl")
    
    # Print final results
    print("\n" + "="*50)
    print("Hyperparameter Tuning Results")
    print("="*50)
    if best_params:
        print(f"Best MAPE: {best_mape:.2%}")
        print("\nBest hyperparameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
    else:
        print("No successful trials completed.")
    
    return best_params, best_mape

def apply_best_params(station_id=None, data_file='merged_station_test.csv', params=None):
    """
    Apply best parameters and train a model.
    
    Args:
        station_id: Station ID to train model for
        data_file: Input data file
        params: Dictionary of parameters to apply
    
    Returns:
        output_dir: Directory where the model is saved
    """
    if not params:
        # Default optimized parameters
        params = {
            'hidden_dim': 384,
            'num_layers': 2,
            'drop_rate': 0.3,
            'batch_size': 128,
            'lr': 5e-4,
            'top_k': 90,
            'use_stl': True,
            'power_weight': 1.2,
            'not_use_power_weight': 0.7,
        }
    
    # Print parameters
    print("\nApplying parameters:")
    for key, value in params.items():
        if key != 'best_mape':
            print(f"  {key}: {value}")
            if key in CFG:
                CFG[key] = value
    
    # Train model with best parameters
    print("\nTraining model with best parameters...")
    output_dir = train_model(station_id, data_file)
    print(f"Model saved to: {output_dir}")
    
    return output_dir

def incremental_finetune(model_dir, station_id=None, data_file='merged_station_test.csv', 
                         epochs=50, lr=1e-5, focus_days=None):
    """
    Incrementally fine-tune a pre-trained model with a focus on specific days.
    Uses the incremental_finetune.py script.
    """
    # Import and use the incremental_finetune function from incremental_finetune.py
    from incremental_finetune import incremental_finetune as inc_finetune
    
    return inc_finetune(model_dir, station_id, data_file, epochs, lr, focus_days)
    
    # Read and process data
    df = pd.read_csv(data_file, parse_dates=['ts'], dtype={'station_ref_id': str})
    df = df.sort_values(['station_ref_id', 'ts'])
    
    # Filter for station
    if station_id:
        df = df[df['station_ref_id'] == station_id].copy()
        if len(df) == 0:
            raise ValueError(f"No data found for station {station_id}")
        print(f"Fine-tuning model for station: {station_id}")
    else:
        # Use first station if none specified
        station_id = df['station_ref_id'].iloc[0]
        df = df[df['station_ref_id'] == station_id].copy()
        print(f"Fine-tuning model for station: {station_id} (first station)")
    
    # Create dataset
    Xp, Xf, Y_power, Y_not_use_power, _, _, _, _ = make_dataset(
        df, config['past_steps'], config['future_steps']
    )
    
    print(f"Samples: {len(Xp)}")
    
    if len(Xp) == 0:
        raise ValueError("No samples generated. Check data length and parameters.")
    
    # Create train/validation split
    spl = int(.8 * len(Xp))
    tr_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Xp[:spl]), 
        torch.from_numpy(Xf[:spl]),
        torch.from_numpy(Y_power[:spl]), 
        torch.from_numpy(Y_not_use_power[:spl])
    )
    va_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Xp[spl:]), 
        torch.from_numpy(Xf[spl:]),
        torch.from_numpy(Y_power[spl:]), 
        torch.from_numpy(Y_not_use_power[spl:])
    )
    
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=config['batch_size'], shuffle=True)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=config['batch_size'], shuffle=False)
    
    # Create custom loss function with focus on specific days
    class FocusedWeightedL1Loss(torch.nn.Module):
        def __init__(self, fut, device, focus_days=None):
            super().__init__()
            # Base weights - higher for days 3-4
            w = np.concatenate([
                np.ones(96*2),           # Day1-2
                np.ones(96)*1.3,         # Day3
                np.ones(96)*1.5,         # Day4
                np.ones(96*3)*1.2        # Day5-7
            ])
            
            # Apply additional focus on specific days
            if focus_days:
                for day in focus_days:
                    if 1 <= day <= 7:
                        start_idx = (day - 1) * 96
                        end_idx = day * 96
                        w[start_idx:end_idx] *= 2.0  # Double the weight for focus days
            
            self.register_buffer('w', torch.tensor(w, dtype=torch.float32, device=device))
            
        def forward(self, pred, target):
            return torch.mean(self.w * torch.abs(pred - target))
    
    # Set up fine-tuning
    model.train()
    crit = FocusedWeightedL1Loss(config['future_steps'], dev, focus_days)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5, factor=.5)
    
    best = 1e9
    wait = 0
    patience = 15
    print("Fine-tuning...")
    
    for ep in range(1, epochs + 1):
        model.train()
        tr = 0
        for xe, xd, yy_power, yy_not_use_power in tr_loader:
            xe, xd, yy_power, yy_not_use_power = xe.to(dev), xd.to(dev), yy_power.to(dev), yy_not_use_power.to(dev)
            opt.zero_grad()
            pred_power, pred_not_use_power = model(xe, xd)
            loss_power = crit(pred_power, yy_power)
            loss_not_use_power = crit(pred_not_use_power, yy_not_use_power)
            loss = config['power_weight'] * loss_power + config['not_use_power_weight'] * loss_not_use_power
            loss.backward()
            opt.step()
            tr += loss.item()
        tr /= len(tr_loader)
        
        model.eval()
        va = 0
        with torch.no_grad():
            for xe, xd, yy_power, yy_not_use_power in va_loader:
                xe, xd, yy_power, yy_not_use_power = xe.to(dev), xd.to(dev), yy_power.to(dev), yy_not_use_power.to(dev)
                pred_power, pred_not_use_power = model(xe, xd)
                loss_power = crit(pred_power, yy_power)
                loss_not_use_power = crit(pred_not_use_power, yy_not_use_power)
                va += (config['power_weight'] * loss_power + config['not_use_power_weight'] * loss_not_use_power).item()
        va /= len(va_loader)
        sch.step(va)
        
        if ep % 5 == 0:
            print(f'E{ep:03d} tr{tr:.4f} va{va:.4f}')
        
        if va < best:
            best = va
            wait = 0
            torch.save(model.state_dict(), f'{output_dir}/model_power.pth')
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break
    
    # Evaluate MAPE for each day
    print("Evaluating MAPE for last 7 days prediction...")
    
    # Load the best model
    model.load_state_dict(torch.load(f'{output_dir}/model_power.pth'))
    model.eval()
    
    all_power_preds = []
    all_not_use_power_preds = []
    all_power_targets = []
    all_not_use_power_targets = []
    
    with torch.no_grad():
        for xe, xd, yy_power, yy_not_use_power in va_loader:
            xe, xd = xe.to(dev), xd.to(dev)
            pred_power, pred_not_use_power = model(xe, xd)
            
            # Convert predictions back to original scale
            pred_power_np = sc_y_power.inverse_transform(pred_power.cpu().numpy())
            pred_not_use_power_np = sc_y_not_use_power.inverse_transform(pred_not_use_power.cpu().numpy())
            
            # Convert targets back to original scale
            target_power_np = sc_y_power.inverse_transform(yy_power.numpy())
            target_not_use_power_np = sc_y_not_use_power.inverse_transform(yy_not_use_power.numpy())
            
            all_power_preds.append(pred_power_np)
            all_not_use_power_preds.append(pred_not_use_power_np)
            all_power_targets.append(target_power_np)
            all_not_use_power_targets.append(target_not_use_power_np)
    
    # Concatenate all predictions and targets
    all_power_preds = np.concatenate(all_power_preds, axis=0)
    all_not_use_power_preds = np.concatenate(all_not_use_power_preds, axis=0)
    all_power_targets = np.concatenate(all_power_targets, axis=0)
    all_not_use_power_targets = np.concatenate(all_not_use_power_targets, axis=0)
    
    # Calculate MAPE for each day
    days = 7
    points_per_day = 96  # 15-minute intervals for 24 hours
    
    power_mape_by_day = []
    not_use_power_mape_by_day = []
    
    for day in range(days):
        start_idx = day * points_per_day
        end_idx = (day + 1) * points_per_day
        
        # Calculate MAPE for total_active_power
        day_power_preds = all_power_preds[:, start_idx:end_idx]
        day_power_targets = all_power_targets[:, start_idx:end_idx]
        # Filter out zeros in targets to avoid division by zero
        mask = day_power_targets > 1.0
        if np.any(mask):
            day_power_mape = mean_absolute_percentage_error(
                day_power_targets[mask], 
                day_power_preds[mask]
            )
            power_mape_by_day.append(day_power_mape)
        else:
            power_mape_by_day.append(np.nan)
        
        # Calculate MAPE for not_use_power
        day_not_use_power_preds = all_not_use_power_preds[:, start_idx:end_idx]
        day_not_use_power_targets = all_not_use_power_targets[:, start_idx:end_idx]
        # Filter out zeros in targets to avoid division by zero
        mask = day_not_use_power_targets > 1.0
        if np.any(mask):
            day_not_use_power_mape = mean_absolute_percentage_error(
                day_not_use_power_targets[mask], 
                day_not_use_power_preds[mask]
            )
            not_use_power_mape_by_day.append(day_not_use_power_mape)
        else:
            not_use_power_mape_by_day.append(np.nan)
    
    # Calculate overall MAPE
    mask = all_power_targets > 1.0
    overall_power_mape = mean_absolute_percentage_error(
        all_power_targets[mask], 
        all_power_preds[mask]
    ) if np.any(mask) else np.nan
    
    mask = all_not_use_power_targets > 1.0
    overall_not_use_power_mape = mean_absolute_percentage_error(
        all_not_use_power_targets[mask], 
        all_not_use_power_preds[mask]
    ) if np.any(mask) else np.nan
    
    # Print MAPE results
    print("\nMAPE Evaluation Results:")
    print("------------------------")
    print("total_active_power MAPE by day:")
    for day, mape in enumerate(power_mape_by_day, 1):
        print(f"  Day {day}: {mape:.2%}")
    print(f"Overall total_active_power MAPE: {overall_power_mape:.2%}")
    
    print("\nnot_use_power MAPE by day:")
    for day, mape in enumerate(not_use_power_mape_by_day, 1):
        print(f"  Day {day}: {mape:.2%}")
    print(f"Overall not_use_power MAPE: {overall_not_use_power_mape:.2%}")
    
    # Save MAPE results
    mape_results = {
        'power_mape_by_day': power_mape_by_day,
        'not_use_power_mape_by_day': not_use_power_mape_by_day,
        'overall_power_mape': overall_power_mape,
        'overall_not_use_power_mape': overall_not_use_power_mape
    }
    joblib.dump(mape_results, f'{output_dir}/mape_results.pkl')
    
    # Save model components
    joblib.dump(sc_e, f'{output_dir}/scaler_enc.pkl')
    joblib.dump(sc_d, f'{output_dir}/scaler_dec.pkl')
    joblib.dump(sc_y_power, f'{output_dir}/scaler_y_power.pkl')
    joblib.dump(sc_y_not_use_power, f'{output_dir}/scaler_y_not_use_power.pkl')
    joblib.dump(ENC_COLS, f'{output_dir}/enc_cols.pkl')
    joblib.dump(DEC_COLS, f'{output_dir}/dec_cols.pkl')
    joblib.dump(config, f'{output_dir}/config.pkl')
    
    # Save station info
    station_info = {
        'station_id': station_id,
        'fine_tuned_from': model_dir,
        'focus_days': focus_days,
        'fine_tuning_epochs': epochs,
        'fine_tuning_lr': lr
    }
    joblib.dump(station_info, f'{output_dir}/station_info.pkl')
    
    print(f"Fine-tuned model saved to {output_dir}/")
    
    return output_dir, overall_power_mape

def main():
    parser = argparse.ArgumentParser(description='Auto-tune power prediction model')
    parser.add_argument('--mode', type=str, required=True, choices=['tune', 'apply', 'finetune'],
                       help='Mode: tune (hyperparameter tuning), apply (apply best params), finetune (fine-tune model)')
    parser.add_argument('--station_id', type=str, default='1716387625733984256',
                       help='Station ID to train model for')
    parser.add_argument('--data_file', type=str, default='merged_station_test.csv', 
                       help='Input data file')
    parser.add_argument('--n_trials', type=int, default=10,
                       help='Number of hyperparameter tuning trials')
    parser.add_argument('--model_dir', type=str,
                       help='Directory containing the pre-trained model (for finetune mode)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of fine-tuning epochs (for finetune mode)')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate for fine-tuning (for finetune mode)')
    parser.add_argument('--focus_days', type=int, nargs='+',
                       help='Days to focus on (1-7) (for finetune mode)')
    parser.add_argument('--random_search', action='store_true',
                       help='Use random search instead of grid search (for tune mode)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'tune':
            # Define parameter grid for tuning
            param_grid = {
                'hidden_dim': [128, 256, 384, 512],
                'num_layers': [1, 2, 3],
                'drop_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
                'batch_size': [64, 128, 256],
                'lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                'top_k': [60, 70, 80, 90, 100],
                'use_stl': [True, False],
                'power_weight': [0.8, 1.0, 1.2, 1.5],
                'not_use_power_weight': [0.5, 0.7, 0.8, 1.0, 1.2]
            }
            
            best_params, best_mape = grid_search(
                param_grid,
                args.station_id,
                args.data_file,
                args.n_trials,
                args.random_search
            )
            
            print(f"Tuning completed successfully!")
            print(f"Best MAPE: {best_mape:.2%}")
            
            # Apply best parameters
            if best_params:
                print("\nTraining final model with best parameters...")
                output_dir = apply_best_params(args.station_id, args.data_file, best_params)
                print(f"Final model saved to: {output_dir}")
        
        elif args.mode == 'apply':
            # Apply default or best parameters
            output_dir = apply_best_params(args.station_id, args.data_file)
            print(f"Model saved to: {output_dir}")
        
        elif args.mode == 'finetune':
            if not args.model_dir:
                print("Error: --model_dir is required for finetune mode")
                sys.exit(1)
            
            output_dir, mape = incremental_finetune(
                args.model_dir,
                args.station_id,
                args.data_file,
                args.epochs,
                args.lr,
                args.focus_days
            )
            
            print(f"Fine-tuning completed successfully!")
            print(f"Final MAPE: {mape:.2%}")
            print(f"Fine-tuned model saved to: {output_dir}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()