import numpy as np
from config.costs_values import cold_storage_cost, warm_storage_cost, hot_storage_cost, cold_operation_cost, warm_operation_cost, hot_operation_cost, cold_retrieval_cost, warm_retrieval_cost, hot_retrieval_cost, cold_latency, warm_latency, hot_latency

def get_online_predictions(data, volumes):
    """
    Classifies test data based on costs and number of accesses.
    
    Args:
        data: Data containing number of accesses
        volumes: Data volumes
    
    Returns:
        predictions: Array with classifications (0=COLD, 1=WARM, 2=HOT)
    """
    
    # Extract number of accesses from test data
    # Calculate total accesses per object (sum of columns)
    if hasattr(data, 'sum'):
        # If DataFrame, sum columns to get total accesses per object
        accesses = data.sum(axis=1).values
    else:
        # If array, assume it's already the number of accesses per object
        accesses = np.array(data)
    
    # Fixed volume for calculations (using mean of volumes if available)
    if volumes is not None:
        fixed_volume = np.mean(volumes) if len(volumes) > 0 else 1.0
    else:
        fixed_volume = 1.0
    
    predictions = []
    
    # For each item in test data
    for i, num_accesses in enumerate(accesses):
        # Calculate costs for the three storage classes
        hot_cost = (hot_storage_cost * fixed_volume) + \
                  (hot_operation_cost * num_accesses * fixed_volume) + \
                  (hot_retrieval_cost * num_accesses * fixed_volume)
        
        warm_cost = (warm_storage_cost * fixed_volume) + \
                   (warm_operation_cost * num_accesses * fixed_volume) + \
                   (warm_retrieval_cost * num_accesses * fixed_volume)
        
        cold_cost = (cold_storage_cost * fixed_volume) + \
                   (cold_operation_cost * num_accesses * fixed_volume) + \
                   (cold_retrieval_cost * num_accesses * fixed_volume)
        
        # Classify based on lowest cost
        costs = [cold_cost, warm_cost, hot_cost]
        optimal_class = np.argmin(costs)
        
        predictions.append(optimal_class)
    
    return np.array(predictions)