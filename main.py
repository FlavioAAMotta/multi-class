import numpy as np
import pandas as pd
import os
import json
import argparse

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from data_loader import DataLoader
from models.classifiers import get_default_classifiers, set_classifier
from models.user_profiles import UserProfile
from models.onl import get_online_predictions

# ============================================================
# Global Constants and Parameters
# ============================================================

# Storage classes
COLD, WARM, HOT = 0, 1, 2

# Access latencies (ms)
T_disk_cold = 1200.0  # Disk access time for COLD
T_disk_warm = 800.0   # Disk access time for WARM
T_disk_hot = 400.0    # Disk access time for HOT

# ============================================================
# Helper Functions
# ============================================================

def apply_cost_weight_thresholds(y_prob, cost_weight, user_profile):
    """
    Applies thresholds based on cost_weight to adjust model predictions.
    
    Args:
        y_prob (array): Probabilities predicted by the model (n_samples, n_classes)
        cost_weight (float): Cost weight (0-1). High values favor cheaper classes (COLD/WARM)
        user_profile (UserProfile): User profile with threshold parameters
    
    Returns:
        array: Adjusted predictions (COLD/WARM/HOT)
    """
    predictions = np.zeros(y_prob.shape[0], dtype=int)
    
    # Calculate adjusted thresholds based on cost_weight
    # high cost_weight = more conservative = favors COLD/WARM
    # low cost_weight = more aggressive = favors HOT
    
    # Base threshold for classifying as HOT (minimum probability)
    base_hot_threshold = 0.4  # 40% probability to classify as HOT
    
    # Adjust threshold based on cost_weight
    # cost_weight = 0.0 (very aggressive) -> hot_threshold = 0.2 (easy to be HOT)
    # cost_weight = 0.5 (neutral) -> hot_threshold = 0.4 (base threshold)
    # cost_weight = 1.0 (very conservative) -> hot_threshold = 0.6 (hard to be HOT)
    hot_threshold = base_hot_threshold + (cost_weight - 0.5) * 0.8
    
    # Threshold for classifying as WARM
    warm_threshold = 0.3 + (cost_weight - 0.5) * 0.4
    
    # Print information about applied thresholds (only first time for each cost_weight)
    current_cost_weight = getattr(apply_cost_weight_thresholds, '_current_cost_weight', None)
    if current_cost_weight != cost_weight:
        print(f"\nApplying thresholds based on cost_weight = {cost_weight:.2f}:")
        print(f"- Threshold for HOT: {hot_threshold:.3f} (high cost_weight = more conservative)")
        print(f"- Threshold for WARM: {warm_threshold:.3f}")
        print(f"- Interpretation: cost_weight = {cost_weight:.2f} {'(conservative)' if cost_weight > 0.5 else '(aggressive)' if cost_weight < 0.5 else '(neutral)'}")
        apply_cost_weight_thresholds._current_cost_weight = cost_weight
    
    # For each sample, apply the threshold based on cost_weight
    for i, probs in enumerate(y_prob):
        # Normalize probabilities to ensure they sum to 1
        probs = probs / np.sum(probs)
        
        # Apply thresholds
        if probs[2] >= hot_threshold:  # HOT probability >= threshold
            predictions[i] = 2  # HOT
        elif probs[1] >= warm_threshold:  # WARM probability >= threshold
            predictions[i] = 1  # WARM
        else:
            predictions[i] = 0  # COLD
    
    return predictions

def calculate_latency(predictions, access_counts, volumes, cost_weight=0.5):
    """
    Calculates total latency for a set of objects based on predictions,
    access counts and volumes.

    Args:
        predictions (array): Predictions for each object (COLD/WARM/HOT).
        access_counts (array): Number of accesses per object.
        volumes (array): Object volumes.
        cost_weight (float): Cost weight to create user profile.

    Returns:
        float: Total latency in milliseconds.
    """
    total_latency = 0
    
    # Create user profile to get latency parameters
    user_profile = UserProfile(cost_weight=cost_weight)

    for pred, access_count in zip(predictions, access_counts):
        if access_count == 0:
            continue
            
        if pred == HOT:
            T_disk = user_profile.hot_latency
        elif pred == WARM:
            T_disk = user_profile.warm_latency
        else:  # COLD
            T_disk = user_profile.cold_latency

        total_latency += access_count * T_disk

    return total_latency

def get_time_windows(data, window_size, step_size):
    """
    Generates temporal window indices for training and testing.

    Args:
        data (DataFrame): Data with columns representing weeks.
        window_size (int): Number of weeks for each window.
        step_size (int): Shift interval between windows.

    Returns:
        list: List of dictionaries with indices for each window.
    """
    windows = []
    num_weeks = len(data.columns)
    total_window_size = window_size * 4  # for training, training label, test and test label
    for start in range(0, num_weeks - total_window_size + 1, step_size):
        windows.append({
            'train': (start, start + window_size),
            'label_train': (start + window_size, start + 2 * window_size),
            'test': (start + window_size, start + 2 * window_size),
            'label_test': (start + 2 * window_size, start + 3 * window_size),
        })
    return windows

def extract_data(df, start, end):
    return df[df.columns[start:end]]

def filter_by_volume(df, df_volume, start_week):
    """
    Filters data based on positive volumes.

    Args:
        df (DataFrame): DataFrame to be filtered.
        df_volume (DataFrame): DataFrame with volume data.
        start_week (int): Week index to be considered.

    Returns:
        DataFrame: Filtered data.
    """
    last_week_column = df_volume.columns[-1]
    valid_objects = df_volume[df_volume[last_week_column] > 0].index
    return df.loc[df.index.intersection(valid_objects)]

def get_label(data, volumes, cost_weight=0.5):
    """
    Generates multi-class labels based on balanced cost calculation for each class.
    
    Args:
        data (DataFrame): Access data
        volumes (Series/Index): Object volumes
        cost_weight (float): Cost weight in balancing (0-1)
    
    Returns:
        Series: Labels (COLD/WARM/HOT)
    """
    MIN_BILLABLE_SIZE = 0.128  # 128KB in GB
    labels = []
    
    # Create user profile to get cost parameters
    user_profile = UserProfile(cost_weight=cost_weight)
    
    # Ensure indices are aligned
    if isinstance(volumes, pd.Series):
        common_indices = data.index.intersection(volumes.index)
        data = data.loc[common_indices]
        volumes = volumes.loc[common_indices]
    else:
        # If volumes is an Index, assume indices are already aligned
        common_indices = data.index
    
    for idx in common_indices:
        row = data.loc[idx]
        volume = volumes[idx] if isinstance(volumes, pd.Series) else volumes[data.index.get_loc(idx)]
        accesses = row.sum()
        billable_volume = max(volume, MIN_BILLABLE_SIZE)
        # Convert volume from bytes to gigabytes
        billable_volume = billable_volume / (1024 ** 3)  # bytes to GB
        billable_volume = 1
        # Calculate cost and latency for each class
        costs = []
        
        # COLD
        cold_storage = billable_volume * user_profile.cold_storage_cost
        cold_operation = accesses * billable_volume * user_profile.cold_operation_cost
        cold_retrieval = accesses * billable_volume * user_profile.cold_retrieval_cost
        costs.append(cold_storage + cold_operation + cold_retrieval)
        
        # WARM
        warm_storage = billable_volume * user_profile.warm_storage_cost
        warm_operation = accesses * billable_volume * user_profile.warm_operation_cost
        warm_retrieval = accesses * billable_volume * user_profile.warm_retrieval_cost
        costs.append(warm_storage + warm_operation + warm_retrieval)

        # HOT
        hot_storage = billable_volume * user_profile.hot_storage_cost
        hot_operation = accesses * billable_volume * user_profile.hot_operation_cost
        hot_retrieval = accesses * billable_volume * user_profile.hot_retrieval_cost
        costs.append(hot_storage + hot_operation + hot_retrieval)
        
        
        # Choose the class with lowest balanced cost
        labels.append(np.argmin(costs))

        if volume == 0:
            print(f"Accesses: {accesses}")
            print(f"Volume: {volume}")
            print(f"Billable Volume: {billable_volume}")
            print(f"Costs: {costs}")
            print(f"Labels: {labels}")

    return pd.Series(labels, index=common_indices)

def calculate_cost(predictions, actuals, volumes, access_counts, cost_weight=0.5):
    """
    Calculates total cost in a robust and consistent manner.
    1. Base cost is from the predicted tier.
    2. A migration penalty is added if the prediction is incorrect.
    """
    MIN_BILLABLE_SIZE = 0.128
    total_cost = 0
    user_profile = UserProfile(cost_weight=cost_weight)
    MIGRATION_COST_PER_GB = 0.01 

    for pred, actual, volume, accesses in zip(predictions, actuals, volumes, access_counts):
        access_1k = float(accesses) / 1000.0
        # NOTE: Line 'billable_volume = 1' should be removed for final results.
        billable_volume = max(volume, MIN_BILLABLE_SIZE) / (1024 ** 3)
        
        # --- 1. Calculate DECISION cost (based on prediction) ---
        storage_cost, operation_cost, retrieval_cost = 0, 0, 0

        if pred == HOT:
            storage_cost = billable_volume * user_profile.hot_storage_cost
            operation_cost = access_1k * user_profile.hot_operation_cost
            retrieval_cost = access_1k * billable_volume * user_profile.hot_retrieval_cost
        elif pred == WARM:
            storage_cost = billable_volume * user_profile.warm_storage_cost
            operation_cost = access_1k * user_profile.warm_operation_cost
            retrieval_cost = access_1k * billable_volume * user_profile.warm_retrieval_cost
        else:  # COLD
            storage_cost = billable_volume * user_profile.cold_storage_cost
            operation_cost = access_1k * user_profile.cold_operation_cost
            retrieval_cost = access_1k * billable_volume * user_profile.cold_retrieval_cost

        if pred != actual:
            storage_cost += billable_volume * user_profile.hot_storage_cost
            operation_cost += access_1k * billable_volume * user_profile.hot_operation_cost

        # Total cost of choice, summed in ONE variable.
        monetary_cost = storage_cost + operation_cost + retrieval_cost
            
        # Accumulate total cost, ensuring each object is counted only ONCE.
        total_cost += monetary_cost
    
    return total_cost

def get_oracle_predictions(access_counts, volumes, cost_weight=0.5):
    """
    Determines the best possible classification (Oracle) based on perfect knowledge of future accesses
    and considering the balance between cost and latency.
    
    Args:
        access_counts (array): Number of accesses per object
        volumes (array): Object volumes
        cost_weight (float): Cost weight in balancing (0-1)
    
    Returns:
        array: Optimal predictions (COLD/WARM/HOT)
    """
    MIN_BILLABLE_SIZE = 0.128  # 128KB in GB
    predictions = np.zeros_like(access_counts, dtype=int)
    
    # Create user profile to get cost parameters
    user_profile = UserProfile(cost_weight=cost_weight)
    
    for i, (accesses, volume) in enumerate(zip(access_counts, volumes)):
        # Adjust volume to minimum billable size
        billable_volume = max(volume, MIN_BILLABLE_SIZE)
        billable_volume = billable_volume / (1024 ** 3)  # bytes to GB
        billable_volume = 1
        
        # Calculate cost and latency for each class
        costs = []
        
        # COLD
        cold_storage = billable_volume * user_profile.cold_storage_cost
        cold_operation = accesses * billable_volume * user_profile.cold_operation_cost
        cold_retrieval = accesses * billable_volume * user_profile.cold_retrieval_cost
        cold_cost = cold_storage + cold_operation + cold_retrieval
        costs.append(cold_cost)
        
        # WARM
        warm_storage = billable_volume * user_profile.warm_storage_cost
        warm_operation = accesses * billable_volume * user_profile.warm_operation_cost
        warm_retrieval = accesses * billable_volume * user_profile.warm_retrieval_cost
        warm_cost = warm_storage + warm_operation + warm_retrieval
        costs.append(warm_cost)
        
        # HOT
        hot_storage = billable_volume * user_profile.hot_storage_cost
        hot_operation = accesses * billable_volume * user_profile.hot_operation_cost
        hot_retrieval = accesses * billable_volume * user_profile.hot_retrieval_cost
        hot_cost = hot_storage + hot_operation + hot_retrieval
        costs.append(hot_cost)
        
        # Choose the class with lowest balanced cost
        predictions[i] = np.argmin(costs)
    
    return predictions

def calculate_age(df_access, window_start):
    """
    Calculates the age of each object in weeks, relative to window_start.
    Age = window_start - first_week_with_access (if >0, else 0)
    """
    ages = {}
    for idx, row in df_access.iterrows():
        first_access = next((i for i, val in enumerate(row) if val > 0), None)
        if first_access is not None:
            age = max(0, window_start - first_access)
        else:
            age = 0  # Never accessed
        ages[idx] = age
    return pd.Series(ages)

def run_analysis(window, user_profile, df_access, df_volume, models_to_run, classifiers, output_dir, window_size):
    """
    Executes analysis for a specific window.
    
    Args:
        window (dict): Dictionary with training and test data
        user_profile (UserProfile): User profile
        df_access (DataFrame): Access data
        df_volume (DataFrame): Volume data
        models_to_run (list): List of models to run
        classifiers (dict): Dictionary with classifiers
        output_dir (str): Output directory
        window_size (int): Window size
    
    Returns:
        dict: Analysis results
    """
    print("\nPreparing data...")
    first_week = window['train'][0]
    train_data = extract_data(df_access, *window['train'])
    test_data = extract_data(df_access, *window['test'])
    label_train_data = extract_data(df_access, *window['label_train'])
    label_test_data = extract_data(df_access, *window['label_test'])

    # First filter by volume
    train_data = filter_by_volume(train_data, df_volume, first_week)
    test_data = filter_by_volume(test_data, df_volume, first_week)
    label_train_data = filter_by_volume(label_train_data, df_volume, first_week)
    label_test_data = filter_by_volume(label_test_data, df_volume, first_week)

    # Then calculate ages only for already filtered objects
    train_ages = calculate_age(df_access, window['train'][0])
    test_ages = calculate_age(df_access, window['test'][0])

    # Filter ages to match indices of already filtered data
    train_ages = train_ages[train_ages.index.isin(train_data.index)]
    test_ages = test_ages[test_ages.index.isin(test_data.index)]

    # Add age column to data
    train_data['age'] = train_ages
    test_data['age'] = test_ages
    
    # Load volume data
    volumes = df_volume.loc[:, df_volume.columns[-1]]
    
    # Generate multi-class labels
    y_train = get_label(label_train_data, volumes, user_profile.cost_weight)
    y_test = get_label(label_test_data, volumes, user_profile.cost_weight)

    # Printing how many objects we have in each class for debugging
    print(f"\nClass distribution (training):")
    print(y_train.value_counts())
    print(f"\nClass distribution (test):")
    print(y_test.value_counts())
    
    print(f"\nClass distribution (training):")
    print(y_train.value_counts(normalize=True))
    print(f"\nClass distribution (test):")
    print(y_test.value_counts(normalize=True))
    
    # Balance training data
    print("\nBalancing training data...")
    
    # Check if there are at least 2 classes to apply SMOTE
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        print(f"WARNING: Only {len(unique_classes)} class(es) found. Skipping balancing.")
        X_train_bal, y_train_bal = train_data, y_train
    else:
        # Calculate number of samples per class
        class_counts = y_train.value_counts()
        min_samples = min(class_counts)
        max_samples = max(class_counts)
        
        # If minority class has less than 10 samples, use a different strategy
        if min_samples < 10:
            print("Using balancing strategy for very imbalanced classes...")
            # Calculate number of neighbors for SMOTE
            n_neighbors = min(5, min_samples - 1)  # Use maximum 5 neighbors or n-1 if n < 5
            
            # Configure SMOTE with adjusted parameters
            smote = SMOTE(
                random_state=42,
                k_neighbors=n_neighbors,
                sampling_strategy={
                    0: max_samples,  # Increase COLD class to size of largest class
                    1: max_samples,  # Increase WARM class to size of largest class
                    2: max_samples   # Keep HOT class as is
                }
            )
        else:
            print("Using standard SMOTE...")
            smote = SMOTE(random_state=42)
        
        X_train_bal, y_train_bal = smote.fit_resample(train_data, y_train)
    
    print("\nDistribution after balancing:")
    print(pd.Series(y_train_bal).value_counts(normalize=True))
    
    print("\nNormalizing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(test_data.values)
    
    access_counts = label_test_data.sum(axis=1)
    volumes = test_data.index.map(lambda x: df_volume.loc[x, df_volume.columns[-1]])
    
    oracle_predictions = get_oracle_predictions(access_counts, volumes, user_profile.cost_weight)
    always_hot_predictions = np.ones_like(y_test) * HOT
    always_warm_predictions = np.ones_like(y_test) * WARM
    always_cold_predictions = np.ones_like(y_test) * COLD
    
    results = {}
    for model_name in models_to_run:
        print("\n" + "-"*50)
        print(f"Treinando modelo: {model_name}")
        print("-"*50)
        
        y_pred = np.zeros_like(y_test)
        
        if model_name == 'AHL':
            y_pred = always_hot_predictions
            print("Applying Always Hot strategy...")
        elif model_name == 'AWL':
            y_pred = always_warm_predictions
            print("Applying Always Warm strategy...")
        elif model_name == 'ACL':
            y_pred = always_cold_predictions
            print("Applying Always Cold strategy...")
        elif model_name == 'ONL':
            print("Applying Online strategy...")
            y_pred = get_online_predictions(test_data, volumes, user_profile.cost_weight)
        else:
            try:
                model = set_classifier(model_name, classifiers)
                print("Starting training...")
                model.fit(X_train_scaled, y_train_bal)
                print("Training completed. Making predictions...")
                
                if model_name == "HV":
                    y_pred = model.predict(X_test_scaled)
                else:
                    y_prob = model.predict_proba(X_test_scaled)
                    y_pred = np.zeros_like(y_test)
                    
                    # Apply thresholds based on cost_weight
                    y_pred = apply_cost_weight_thresholds(y_prob, user_profile.cost_weight, user_profile)
                        
            except Exception as e:
                print(f"ERROR training/predicting with {model_name}: {str(e)}")
                continue
        
        print("\nCalculating metrics...")
        access_counts = label_test_data.sum(axis=1)
        volumes = test_data.index.map(lambda x: df_volume.loc[x, df_volume.columns[-1]])
        
        # Calculate multi-class metrics
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results[model_name] = {
            'cost': calculate_cost(y_pred, y_test, volumes, access_counts, user_profile.cost_weight),
            'latency': calculate_latency(y_pred, access_counts, volumes, user_profile.cost_weight),
            'oracle_cost': calculate_cost(y_test, y_test, volumes, access_counts, user_profile.cost_weight),
            'oracle_latency': calculate_latency(y_test, access_counts, volumes, user_profile.cost_weight),
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"\nResults for {model_name}:")
        print(f"- Cost: {results[model_name]['cost']:,.2f}")
        print(f"- Latency: {results[model_name]['latency']:,.2f} ms")
        print(f"- Precision: {precision:.4f}")
        print(f"- Recall: {recall:.4f}")
        print(f"- F1-Score: {f1:.4f}")
        print(f"- Confusion Matrix:")
        print(cm)
    
    return results

def save_final_results(final_results, output_dir):
    """
    Saves final results to a CSV file.

    Args:
        final_results (dict): Consolidated results.
        output_dir (str): Output directory.
    """
    results_for_csv = []
    for profile_key, results in final_results.items():
        # Extract model_name and cost_weight from key (format: "model_name_cost_weight")
        parts = profile_key.split('_')
        if len(parts) >= 2:
            # Take last element as cost_weight and rest as model_name
            cost_weight = float(parts[-1])
            model_name = '_'.join(parts[:-1])
        else:
            # Fallback for old format
            model_name, cost_weight = profile_key.split('_')
            cost_weight = float(cost_weight)
        cm = results['confusion_matrix']
        
        # Extract metrics from confusion matrix for each class
        cm_shape = cm.shape
        if cm_shape == (2, 2):
            # 2x2 matrix - only 2 classes present
            cold_correct = cm[0][0]
            cold_to_warm = cm[0][1]
            cold_to_hot = 0
            warm_to_cold = cm[1][0]
            warm_correct = cm[1][1]
            warm_to_hot = 0
            hot_to_cold = 0
            hot_to_warm = 0
            hot_correct = 0
        elif cm_shape == (3, 3):
            # 3x3 matrix - all 3 classes present
            cold_correct = cm[0][0]
            cold_to_warm = cm[0][1]
            cold_to_hot = cm[0][2]
            warm_to_cold = cm[1][0]
            warm_correct = cm[1][1]
            warm_to_hot = cm[1][2]
            hot_to_cold = cm[2][0]
            hot_to_warm = cm[2][1]
            hot_correct = cm[2][2]
        else:
            # Unexpected format - use zeros
            cold_correct = cold_to_warm = cold_to_hot = 0
            warm_to_cold = warm_correct = warm_to_hot = 0
            hot_to_cold = hot_to_warm = hot_correct = 0
        
        result_row = {
            'model_name': model_name,
            'cost_weight': cost_weight,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1'],
            'model_cost': results['model_cost'],
            'oracle_cost': results['oracle_cost'],
            'model_latency': results['model_latency'] / 1000,  # Convertendo para segundos
            'oracle_latency': results['oracle_latency'] / 1000,
            'cold_correct': cold_correct,
            'cold_to_warm': cold_to_warm,
            'cold_to_hot': cold_to_hot,
            'warm_to_cold': warm_to_cold,
            'warm_correct': warm_correct,
            'warm_to_hot': warm_to_hot,
            'hot_to_cold': hot_to_cold,
            'hot_to_warm': hot_to_warm,
            'hot_correct': hot_correct
        }
        results_for_csv.append(result_row)

    results_df = pd.DataFrame(results_for_csv)
    output_csv_path = os.path.join(output_dir, 'final_results.csv')
    results_df.to_csv(output_csv_path, index=False)
    print(f"Final results saved to CSV: {output_csv_path}")

def accumulate_results(final_results, window_results, cost_weight):
    """
    Accumulates window results into final results.

    Args:
        final_results (dict): Dictionary with accumulated results.
        window_results (dict): Current window results.
        cost_weight (float): Cost weight used in analysis.
    """
    for model_name, result in window_results.items():
        profile_key = f"{model_name}_{cost_weight:.1f}"
        
        if profile_key not in final_results:
            # Initialize accumulators for the model
            final_results[profile_key] = {
                'model_name': model_name,
                'cost_weight': cost_weight,
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'model_cost': 0,
                'oracle_cost': 0,
                'model_latency': 0,
                'oracle_latency': 0,
                'confusion_matrix': np.zeros((3, 3)),  # 3x3 matrix for three classes
                'n_windows': 0
            }
        
        # Extract y_true and y_pred from confusion matrix
        cm = result['confusion_matrix']
        y_true = []
        y_pred = []
        
        # Check confusion matrix size
        cm_shape = cm.shape
        if cm_shape == (2, 2):
            # 2x2 matrix - only 2 classes present
            for i in range(2):
                for j in range(2):
                    y_true.extend([i] * cm[i][j])
                    y_pred.extend([j] * cm[i][j])
        elif cm_shape == (3, 3):
            # 3x3 matrix - all 3 classes present
            for i in range(3):
                for j in range(3):
                    y_true.extend([i] * cm[i][j])
                    y_pred.extend([j] * cm[i][j])
        else:
            print(f"WARNING: Confusion matrix with unexpected format: {cm_shape}")
            continue
        
        # Calculate metrics for current window
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Update accumulators
        n = final_results[profile_key]['n_windows']
        final_results[profile_key]['model_cost'] += result['cost']
        final_results[profile_key]['oracle_cost'] += result['oracle_cost']
        final_results[profile_key]['model_latency'] += result['latency']
        final_results[profile_key]['oracle_latency'] += result['oracle_latency']
        
        # Sum confusion matrix considering different sizes
        cm_shape = cm.shape
        stored_cm = final_results[profile_key]['confusion_matrix']
        
        if cm_shape == (2, 2) and stored_cm.shape == (3, 3):
            # Expand 2x2 matrix to 3x3
            expanded_cm = np.zeros((3, 3))
            expanded_cm[:2, :2] = cm
            final_results[profile_key]['confusion_matrix'] += expanded_cm
        elif cm_shape == (3, 3) and stored_cm.shape == (3, 3):
            # Both are 3x3
            final_results[profile_key]['confusion_matrix'] += cm
        elif cm_shape == (2, 2) and stored_cm.shape == (2, 2):
            # Both are 2x2
            final_results[profile_key]['confusion_matrix'] += cm
        else:
            print(f"WARNING: Confusion matrix incompatibility: {cm_shape} vs {stored_cm.shape}")
        
        final_results[profile_key]['n_windows'] += 1
        
        # Update metrics with moving average
        final_results[profile_key]['accuracy'] = ((n * final_results[profile_key]['accuracy'] + acc) / (n + 1))
        final_results[profile_key]['precision'] = ((n * final_results[profile_key]['precision'] + prec) / (n + 1))
        final_results[profile_key]['recall'] = ((n * final_results[profile_key]['recall'] + rec) / (n + 1))
        final_results[profile_key]['f1'] = ((n * final_results[profile_key]['f1'] + f1) / (n + 1))

def load_configuration(config_path):
    """Loads framework configuration."""
    print("=== Starting framework execution ===")
    print("\nLoading data and parameters...")
    
    data_loader = DataLoader(data_dir='data/', config_path=config_path)
    params = data_loader.params
    
    print(f"Loaded configurations:")
    print(f"- Configuration file: {config_path}")
    print(f"- Population: {params['pop_name']}")
    print(f"- Window size: {params['window_size']}")
    print(f"- Step: {params['step_size']}")
    print(f"- Models: {params['models_to_run']}")
    
    return data_loader, params

def initialize_components(params):
    """Initializes classifiers and loads data."""
    print("\nInitializing classifiers...")
    classifiers = get_default_classifiers()

    print("\nLoading access and volume data...")
    data_loader = DataLoader(data_dir='data/', config_path='config/config.yaml')
    df_access = data_loader.load_access_data(params['pop_name'])
    df_volume = data_loader.load_volume_data(params['pop_name'])
    df_access.set_index('NameSpace', inplace=True)
    df_volume.set_index('NameSpace', inplace=True)
    
    print(f"Data dimensions:")
    print(f"- Access: {df_access.shape}")
    print(f"- Volumes: {df_volume.shape}")
    
    return classifiers, df_access, df_volume

def setup_output_directories(pop_name, window_size, step_size):
    """Sets up output directories for analyses."""
    output_dir = os.path.join('results', f'results_{pop_name}_{window_size}_{step_size}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def process_window(window, window_index, total_windows, user_profile, df_access, df_volume, 
                  models_to_run, classifiers, output_dir, window_size):
    """Processes a specific window"""
    print(f"\n=== Processing window {window_index+1}/{total_windows} ===")

    print("\n--- Executing analysis ---")
    results = run_analysis(window, user_profile, df_access, df_volume, 
                                   models_to_run, classifiers, output_dir, 
                                   window_size)
    
    return results

def setup_experiment_parameters(params):
    """Sets up experiment parameters."""
    window_size = params['window_size']
    step_size = params['step_size']
    models_to_run = params['models_to_run']
    cost_weights = params.get('cost_weight', 0.5)
    
    # Ensure cost_weights is a list
    if not isinstance(cost_weights, list):
        cost_weights = [cost_weights]
    
    return window_size, step_size, models_to_run, cost_weights

def generate_time_windows(df_access, window_size, step_size):
    """Generates temporal windows for the experiment."""
    windows = get_time_windows(df_access, window_size, step_size)
    print(f"Generated {len(windows)} temporal windows for processing")
    return windows

def initialize_result_containers():
    """Initializes containers to store results."""
    final_results = {}
    return final_results

def accumulate_window_results(final_results,  
                            results, cost_weight):
    """Accumulates window results into final containers."""
    accumulate_results(final_results, results, cost_weight)

def process_all_windows(df_access, df_volume, params, classifiers, output_dir):
    """Processes all temporal windows."""
    print("\nStarting window processing...")
    
    # Set up experiment parameters
    window_size, step_size, models_to_run, cost_weights = setup_experiment_parameters(params)
    
    # Generate temporal windows
    windows = generate_time_windows(df_access, window_size, step_size)
    
    # Initialize result containers
    final_results = initialize_result_containers()
    
    # Process each cost_weight value
    for cost_weight in cost_weights:
        print(f"\n=== Processing with cost_weight = {cost_weight} ===")
        user_profile = UserProfile(cost_weight)
        
        # Process each window for this cost_weight
        for i, window in enumerate(windows):
            results = process_window(
                window, i, len(windows), user_profile, df_access, df_volume,
                models_to_run, classifiers, output_dir, window_size
            )
            
            # Accumulate results
            accumulate_window_results(
                final_results,
                results, cost_weight
            )
    
    return final_results, len(windows) * len(cost_weights)

def save_results(final_results, output_dir, total_windows):
    """Saves final results."""
    save_final_results(final_results, output_dir)
    print(f"Total windows processed: {total_windows}")

def main():
    """Main function that executes the multi-class classification framework."""
    # Argument parser configuration
    parser = argparse.ArgumentParser(description='Multi-class classification framework for storage')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file (default: config/config.yaml)')
    
    args = parser.parse_args()
    
    # Load configuration
    data_loader, params = load_configuration(args.config)
    
    # Initialize components
    classifiers, df_access, df_volume = initialize_components(params)
    
    # Set up output directories
    output_dir = setup_output_directories(
        params['pop_name'], params['window_size'], params['step_size']
    )
    
    # Process all windows
    final_results, total_windows = process_all_windows(
        df_access, df_volume, params, classifiers, output_dir
    )
    
    # Save results
    save_results(final_results, output_dir, total_windows)

if __name__ == '__main__':
    main()
