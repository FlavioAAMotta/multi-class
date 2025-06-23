from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from config.costs_values import (
    cold_storage_cost, warm_storage_cost, hot_storage_cost, 
    cold_operation_cost, warm_operation_cost, hot_operation_cost, 
    cold_retrieval_cost, warm_retrieval_cost, hot_retrieval_cost,
    cold_latency, warm_latency, hot_latency
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    
def apply_smote_balancing(train_data, y_train):
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

    # Verifica se há mais de uma classe
    unique_classes = y_train.unique()
    if len(unique_classes) < 2:
        print(f"Aviso: Apenas {len(unique_classes)} classe(s) encontrada(s). Retornando dados originais.")
        return train_data, y_train

    # Calcula o número de amostras por classe
    class_counts = y_train.value_counts()
    min_samples = min(class_counts)
    max_samples = max(class_counts)

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
        Series: Rótulos (0=COLD, 1=WARM, 2=HOT)
    """
    MIN_BILLABLE_SIZE = 0.128  # 128KB em GB
    labels = []
    
    print(f"Gerando rótulos para {len(data)} registros")
    
    # Garante que os índices estejam alinhados
    if isinstance(volumes, pd.Series):
        common_indices = data.index.intersection(volumes.index)
        data = data.loc[common_indices]
        volumes = volumes.loc[common_indices]
    else:
        # Se volumes for um Index, assume que os índices já estão alinhados
        common_indices = data.index
    
    print(f"Índices comuns: {len(common_indices)}")
    
    for idx in common_indices:
        try:
            row = data.loc[idx]
            volume = volumes[idx] if isinstance(volumes, pd.Series) else volumes[data.index.get_loc(idx)]
            accesses = row.sum()
            billable_volume = max(volume, MIN_BILLABLE_SIZE)
            
            # Calcula custo total para cada classe (armazenamento + operação + recuperação)
            costs = []
            
            # HOT (0=COLD, 1=WARM, 2=HOT)
            hot_storage = billable_volume * hot_storage_cost
            hot_operation = accesses * billable_volume * hot_operation_cost
            hot_retrieval = accesses * billable_volume * hot_retrieval_cost
            hot_total_cost = hot_storage + hot_operation + hot_retrieval
            costs.append(hot_total_cost)
            
            # WARM
            warm_storage = billable_volume * warm_storage_cost
            warm_operation = accesses * billable_volume * warm_operation_cost
            warm_retrieval = accesses * billable_volume * warm_retrieval_cost
            warm_total_cost = warm_storage + warm_operation + warm_retrieval
            costs.append(warm_total_cost)
            
            # COLD
            cold_storage = billable_volume * cold_storage_cost
            cold_operation = accesses * billable_volume * cold_operation_cost
            cold_retrieval = accesses * billable_volume * cold_retrieval_cost
            cold_total_cost = cold_storage + cold_operation + cold_retrieval
            costs.append(cold_total_cost)
            
            # Aplica o peso do custo para balancear com latência
            # Se cost_weight é alto, favorece classes mais baratas (COLD/WARM)
            # Se cost_weight é baixo, favorece classes mais rápidas (HOT)
            if cost_weight > 0.7:  # Conservador - favorece custo baixo
                costs[2] *= 1.2  # Penaliza HOT
                costs[1] *= 1.1  # Penaliza WARM levemente
            elif cost_weight < 0.3:  # Agressivo - favorece performance
                costs[0] *= 1.2  # Penaliza COLD
                costs[1] *= 1.1  # Penaliza WARM levemente
            
            # Escolhe a classe com menor custo balanceado
            optimal_class = np.argmin(costs)
            labels.append(optimal_class)
            
        except Exception as e:
            print(f"Erro ao processar índice {idx}: {e}")
            labels.append(0)  # Default para COLD
    
    print(f"Rótulos gerados: {len(labels)}")
    print(f"Distribuição de classes: {pd.Series(labels).value_counts().to_dict()}")
    return pd.Series(labels, index=common_indices)

def extract_data(df_access, volumes, start, end):
    """
    Extrai dados de acesso para uma janela específica, considerando apenas objetos com volume > 0.
    
    Args:
        df_access (DataFrame): Dados de acesso
        volumes (Series): Volumes dos objetos
        start (int): Índice inicial da janela
        end (int): Índice final da janela
        
    Returns:
        DataFrame: Dados de acesso filtrados para a janela
    """
    # Verifica se há objetos com volume > 0 na janela
    exists_in_time_window = (df_access.iloc[:, start:end] > 0).any(axis=1)
    
    # Filtra os objetos que existem na janela
    filtered_access = df_access.loc[exists_in_time_window]
    
    # Seleciona apenas as colunas da janela específica
    window_columns = df_access.columns[start:end]
    result = filtered_access[window_columns]
    
    print(f"Extraindo dados da janela {start}:{end}, objetos encontrados: {len(result)}")
    return result

def run_analysis(window, df_access, df_volume, models_to_run, classifiers, output_dir, window_size):
    print(f"Debug: Iniciando run_analysis com janela {window}")
    print(f"Debug: df_access shape: {df_access.shape}")
    print(f"Debug: df_volume shape: {df_volume.shape}")
    
    volumes = df_volume.loc[:, df_volume.columns[-1]]
    print(f"Debug: volumes shape: {volumes.shape}")

    # Extrair dados de treino e teste
    print("Debug: Extraindo dados de treino...")
    train_data = extract_data(df_access, volumes, *window['train'])
    print("Debug: Extraindo dados de teste...")
    test_data = extract_data(df_access, volumes, *window['test'])
    print("Debug: Extraindo dados de rótulo treino...")
    label_train_data = extract_data(df_access, volumes, *window['label_train'])
    print("Debug: Extraindo dados de rótulo teste...")
    label_test_data = extract_data(df_access, volumes, *window['label_test'])

    # Verificar se há dados suficientes
    if len(train_data) == 0 or len(test_data) == 0:
        print("Aviso: Dados insuficientes para esta janela. Pulando...")
        return {}

    # Encontrar objetos comuns entre dados de treino e rótulos de treino
    common_train_objects = train_data.index.intersection(label_train_data.index)
    common_test_objects = test_data.index.intersection(label_test_data.index)
    
    print(f"Debug: Objetos comuns treino: {len(common_train_objects)}")
    print(f"Debug: Objetos comuns teste: {len(common_test_objects)}")
    
    # Filtrar para usar apenas objetos comuns
    train_data = train_data.loc[common_train_objects]
    label_train_data = label_train_data.loc[common_train_objects]
    test_data = test_data.loc[common_test_objects]
    label_test_data = label_test_data.loc[common_test_objects]

    # Gera rótulos multiclasse
    print("Debug: Gerando rótulos de treino...")
    cost_weight = 0.5  # Valor padrão, pode ser obtido do config se necessário
    y_train = get_label(label_train_data, volumes, cost_weight)
    print("Debug: Gerando rótulos de teste...")
    y_test = get_label(label_test_data, volumes, cost_weight)

    # Verificar se há rótulos válidos
    if len(y_train) == 0 or len(y_test) == 0:
        print("Aviso: Rótulos insuficientes para esta janela. Pulando...")
        return {}

    # Verificar se há pelo menos duas classes nos dados de teste
    if len(y_test.unique()) < 2:
        print(f"Aviso: Apenas {len(y_test.unique())} classe(s) nos dados de teste. Pulando...")
        return {}

    print("Debug: Aplicando SMOTE...")
    X_train_bal, y_train_bal = apply_smote_balancing(train_data, y_train)

    resultados = {}

    # Treina os modelos
    print("Debug: Treinando modelos...")
    for model_name in models_to_run:
        if model_name in classifiers:
            model = classifiers[model_name]
            try:
                model.fit(X_train_bal, y_train_bal)
                print(f"Modelo {model_name} treinado com sucesso")
            except Exception as e:
                print(f"Erro ao treinar modelo {model_name}: {e}")
                continue
        else:
            print(f"Modelo {model_name} não encontrado nos classificadores disponíveis")
            continue

    # Testa os modelos
    print("Debug: Testando modelos...")
    for model_name in models_to_run:
        if model_name not in classifiers:
            continue
            
        model = classifiers[model_name]
        try:
            y_pred = model.predict(test_data)
            
            # Verificar se as predições têm pelo menos duas classes
            if len(np.unique(y_pred)) < 2:
                print(f"Aviso: Modelo {model_name} prediz apenas uma classe. Pulando métricas...")
                continue
            
            metricas = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='macro', zero_division=0)
            }
            
            resultados[model_name] = metricas
            
            print(f"Model: {model_name}")
            print(f"Accuracy: {metricas['accuracy']:.4f}")
            print(f"Precision: {metricas['precision']:.4f}")
            print(f"Recall: {metricas['recall']:.4f}")
            print(f"F1 Score: {metricas['f1']:.4f}")
            
        except Exception as e:
            print(f"Erro ao testar modelo {model_name}: {e}")
            continue

    return resultados