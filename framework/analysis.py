from models.user_profiles import UserProfile
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np


    
def apply_smote_balancing(train_data, y_train, min_samples, max_samples):
    """
    Aplica balanceamento SMOTE aos dados de treino.
    
    Args:
        train_data: Dados de treino
        y_train: Rótulos de treino
        min_samples: Número mínimo de amostras por classe
        max_samples: Número máximo de amostras por classe
        
    Returns:
        tuple: Dados e rótulos balanceados (X_train_bal, y_train_bal)
    """
    # Se a classe minoritária tiver menos de 10 amostras, usamos uma estratégia diferente
    if min_samples < 10:
        print("Usando estratégia de balanceamento para classes muito desbalanceadas...")
        # Calcula o número de vizinhos para o SMOTE
        n_neighbors = min(5, min_samples - 1)  # Usa no máximo 5 vizinhos ou n-1 se n < 5
        
        # Configura o SMOTE com parâmetros ajustados
        smote = SMOTE(
            random_state=42,
            k_neighbors=n_neighbors,
            sampling_strategy={
                0: max_samples,  # Aumenta a classe COLD para o tamanho da maior classe
                1: max_samples,  # Aumenta a classe WARM para o tamanho da maior classe
                2: max_samples   # Mantém a classe HOT como está
            }
        )
    else:
        print("Usando SMOTE padrão...")
        smote = SMOTE(random_state=42)
    
    X_train_bal, y_train_bal = smote.fit_resample(train_data, y_train)
    
    print("\nDistribuição após balanceamento:")
    print(pd.Series(y_train_bal).value_counts(normalize=True))
    
    return X_train_bal, y_train_bal
    
def get_label(data, volumes, cost_weight=0.5):
    """
    Gera rótulos multiclasse baseado no cálculo do custo balanceado para cada classe.
    
    Args:
        data (DataFrame): Dados de acesso
        volumes (Series/Index): Volumes dos objetos
        cost_weight (float): Peso do custo no balanceamento (0-1)
    
    Returns:
        Series: Rótulos (COLD/WARM/HOT)
    """
    MIN_BILLABLE_SIZE = 0.128  # 128KB em GB
    labels = []
    
    # Cria perfil do usuário para obter os parâmetros de custo
    user_profile = UserProfile(cost_weight=cost_weight)
    
    # Garante que os índices estejam alinhados
    if isinstance(volumes, pd.Series):
        common_indices = data.index.intersection(volumes.index)
        data = data.loc[common_indices]
        volumes = volumes.loc[common_indices]
    else:
        # Se volumes for um Index, assume que os índices já estão alinhados
        common_indices = data.index
    
    for idx in common_indices:
        row = data.loc[idx]
        volume = volumes[idx] if isinstance(volumes, pd.Series) else volumes[data.index.get_loc(idx)]
        accesses = row.sum()
        billable_volume = max(volume, MIN_BILLABLE_SIZE)
        
        # Calcula custo e latência para cada classe
        costs = []
        
        # HOT
        hot_storage = billable_volume * user_profile.hot_storage_cost
        hot_operation = accesses * billable_volume * user_profile.hot_operation_cost
        hot_latency = accesses * user_profile.hot_latency
        hot_cost = (cost_weight * (hot_storage + hot_operation)) + \
                  ((1 - cost_weight) * (hot_latency / (user_profile.cold_latency * accesses) if accesses > 0 else 0))
        costs.append(hot_cost)
        
        # WARM
        warm_storage = billable_volume * user_profile.warm_storage_cost
        warm_operation = accesses * billable_volume * user_profile.warm_operation_cost
        warm_retrieval = accesses * billable_volume * user_profile.warm_retrieval_cost
        warm_latency = accesses * user_profile.warm_latency
        warm_cost = (cost_weight * (warm_storage + warm_operation + warm_retrieval)) + \
                   ((1 - cost_weight) * (warm_latency / (user_profile.cold_latency * accesses) if accesses > 0 else 0))
        costs.append(warm_cost)
        
        # COLD
        cold_storage = billable_volume * user_profile.cold_storage_cost
        cold_operation = accesses * billable_volume * user_profile.cold_operation_cost
        cold_retrieval = accesses * billable_volume * user_profile.cold_retrieval_cost
        cold_latency = accesses * user_profile.cold_latency
        cold_cost = (cost_weight * (cold_storage + cold_operation + cold_retrieval)) + \
                   ((1 - cost_weight) * (cold_latency / (user_profile.cold_latency * accesses) if accesses > 0 else 0))
        costs.append(cold_cost)
        
        # Escolhe a classe com menor custo balanceado
        labels.append(np.argmin(costs))
    
    return pd.Series(labels, index=common_indices)

def extract_data(df_access, volumes, start, end):
    """
    Extracts access data for a specific period. Line by line, if the volume is greater than 0, extracts the access data for the specified period.
    
    Args:
        df_access (DataFrame): Access data
        volumes (Series/Index): Object volumes 
        start (int): Initial index
        end (int): Final index
        
    Returns:
        DataFrame: Access data for the specified period
    """
    data = []
    for i in range(start, end):
        if volumes.iloc[i] > 0:
            data.append(df_access.iloc[i])
    return pd.DataFrame(data)

def run_analysis(window, df_access, df_volume, models_to_run, classifiers, output_dir, window_size):
    volumes = df_volume.loc[:, df_volume.columns[-1]]

    # Extrair dados de treino e teste
    train_data = extract_data(df_access, volumes, *window['train'])
    test_data = extract_data(df_access, volumes, *window['test'])
    label_train_data = extract_data(df_access, volumes, *window['label_train'])
    label_test_data = extract_data(df_access, volumes, *window['label_test'])

    # Gera rótulos multiclasse
    y_train = get_label(label_train_data, volumes, user_profile.cost_weight)
    y_test = get_label(label_test_data, volumes, user_profile.cost_weight)

    # Balancear dados
    train_data, label_train_data = balance_data(train_data, label_train_data)
    test_data, label_test_data = balance_data(test_data, label_test_data)

    # Normalizar dados
    train_data, label_train_data = normalize_data(train_data, label_train_data)

    # Calcula o número de amostras por classe
    class_counts = y_train.value_counts()
    min_samples = min(class_counts)
    max_samples = max(class_counts)

    X_train_bal, y_train_bal = apply_smote_balancing(train_data, y_train, min_samples, max_samples)