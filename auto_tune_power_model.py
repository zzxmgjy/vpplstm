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