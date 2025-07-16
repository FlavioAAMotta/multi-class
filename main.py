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
# Constantes e Parâmetros Globais
# ============================================================

# Classes de armazenamento
COLD, WARM, HOT = 0, 1, 2

# Latências de acesso (ms)
T_disk_cold = 1200.0  # Tempo de acesso ao disco para COLD
T_disk_warm = 800.0   # Tempo de acesso ao disco para WARM
T_disk_hot = 400.0    # Tempo de acesso ao disco para HOT

# ============================================================
# Funções Auxiliares
# ============================================================

def apply_cost_weight_thresholds(y_prob, cost_weight, user_profile):
    """
    Aplica thresholds baseados no cost_weight para ajustar as predições dos modelos.
    
    Args:
        y_prob (array): Probabilidades preditas pelo modelo (n_samples, n_classes)
        cost_weight (float): Peso do custo (0-1). Valores altos favorecem classes mais baratas (COLD/WARM)
        user_profile (UserProfile): Perfil do usuário com parâmetros de threshold
    
    Returns:
        array: Predições ajustadas (COLD/WARM/HOT)
    """
    predictions = np.zeros(y_prob.shape[0], dtype=int)
    
    # Calcula thresholds ajustados baseado no cost_weight
    # cost_weight alto = mais conservador = favorece COLD/WARM
    # cost_weight baixo = mais agressivo = favorece HOT
    
    # Threshold base para classificar como HOT (probabilidade mínima)
    base_hot_threshold = 0.4  # 40% de probabilidade para classificar como HOT
    
    # Ajusta threshold baseado no cost_weight
    # cost_weight = 0.0 (muito agressivo) -> hot_threshold = 0.2 (fácil ser HOT)
    # cost_weight = 0.5 (neutro) -> hot_threshold = 0.4 (threshold base)
    # cost_weight = 1.0 (muito conservador) -> hot_threshold = 0.6 (difícil ser HOT)
    hot_threshold = base_hot_threshold + (cost_weight - 0.5) * 0.8
    
    # Threshold para classificar como WARM
    warm_threshold = 0.3 + (cost_weight - 0.5) * 0.4
    
    # Imprime informações sobre os thresholds aplicados (apenas na primeira vez para cada cost_weight)
    current_cost_weight = getattr(apply_cost_weight_thresholds, '_current_cost_weight', None)
    if current_cost_weight != cost_weight:
        print(f"\nAplicando thresholds baseados no cost_weight = {cost_weight:.2f}:")
        print(f"- Threshold para HOT: {hot_threshold:.3f} (cost_weight alto = mais conservador)")
        print(f"- Threshold para WARM: {warm_threshold:.3f}")
        print(f"- Interpretação: cost_weight = {cost_weight:.2f} {'(conservador)' if cost_weight > 0.5 else '(agressivo)' if cost_weight < 0.5 else '(neutro)'}")
        apply_cost_weight_thresholds._current_cost_weight = cost_weight
    
    # Para cada amostra, aplica o threshold baseado no cost_weight
    for i, probs in enumerate(y_prob):
        # Normaliza as probabilidades para garantir que somem 1
        probs = probs / np.sum(probs)
        
        # Aplica os thresholds
        if probs[2] >= hot_threshold:  # Probabilidade de HOT >= threshold
            predictions[i] = 2  # HOT
        elif probs[1] >= warm_threshold:  # Probabilidade de WARM >= threshold
            predictions[i] = 1  # WARM
        else:
            predictions[i] = 0  # COLD
    
    return predictions

def calculate_latency(predictions, access_counts, volumes, cost_weight=0.5):
    """
    Calcula a latência total para um conjunto de objetos com base nas predições,
    contagens de acesso e volumes.

    Args:
        predictions (array): Predições para cada objeto (COLD/WARM/HOT).
        access_counts (array): Número de acessos por objeto.
        volumes (array): Volumes dos objetos.
        cost_weight (float): Peso do custo para criar o perfil do usuário.

    Returns:
        float: Latência total em milissegundos.
    """
    total_latency = 0
    
    # Cria perfil do usuário para obter os parâmetros de latência
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
    Gera índices de janelas temporais para treinamento e teste.

    Args:
        data (DataFrame): Dados com colunas representando semanas.
        window_size (int): Número de semanas para cada janela.
        step_size (int): Intervalo de deslocamento entre janelas.

    Returns:
        list: Lista de dicionários com os índices de cada janela.
    """
    windows = []
    num_weeks = len(data.columns)
    total_window_size = window_size * 4  # para treino, rótulo treino, teste e rótulo teste
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
    Filtra os dados com base em volumes positivos.

    Args:
        df (DataFrame): DataFrame a ser filtrado.
        df_volume (DataFrame): DataFrame com dados de volume.
        start_week (int): Índice da semana a ser considerado.

    Returns:
        DataFrame: Dados filtrados.
    """
    last_week_column = df_volume.columns[-1]
    valid_objects = df_volume[df_volume[last_week_column] > 0].index
    return df.loc[df.index.intersection(valid_objects)]

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
        # Converte volume de bytes para gigabytes
        billable_volume = billable_volume / (1024 ** 3)  # bytes para GB
        billable_volume = 1
        # Calcula custo e latência para cada classe
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
        
        
        # Escolhe a classe com menor custo balanceado
        labels.append(np.argmin(costs))

        if volume == 0:
            print(f"Acessos: {accesses}")
            print(f"Volume: {volume}")
            print(f"Billable Volume: {billable_volume}")
            print(f"Costs: {costs}")
            print(f"Labels: {labels}")

    return pd.Series(labels, index=common_indices)

def calculate_cost(predictions, actuals, volumes, access_counts, cost_weight=0.5):
    """
    Calcula o custo total de forma robusta e consistente.
    1. O custo base é o da camada predita.
    2. Uma penalidade de migração é adicionada se a predição for incorreta.
    """
    MIN_BILLABLE_SIZE = 0.128
    total_cost = 0
    user_profile = UserProfile(cost_weight=cost_weight)
    MIGRATION_COST_PER_GB = 0.01 

    for pred, actual, volume, accesses in zip(predictions, actuals, volumes, access_counts):
        access_1k = float(accesses) / 1000.0
        # ATENÇÃO: Linha 'billable_volume = 1' deve ser removida para resultados finais.
        billable_volume = max(volume, MIN_BILLABLE_SIZE) / (1024 ** 3)
        
        # --- 1. Calcular o custo da DECISÃO (baseado na predição) ---
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

        # Custo total da escolha, somado em UMA variável.
        monetary_cost = storage_cost + operation_cost + retrieval_cost
            
        # Acumula o custo total, garantindo que cada objeto seja contado apenas UMA vez.
        total_cost += monetary_cost
    
    return total_cost

def get_oracle_predictions(access_counts, volumes, cost_weight=0.5):
    """
    Determina a melhor classificação possível (Oracle) com base no conhecimento perfeito dos acessos futuros
    e considerando o balanceamento entre custo e latência.
    
    Args:
        access_counts (array): Número de acessos por objeto
        volumes (array): Volumes dos objetos
        cost_weight (float): Peso do custo no balanceamento (0-1)
    
    Returns:
        array: Predições ótimas (COLD/WARM/HOT)
    """
    MIN_BILLABLE_SIZE = 0.128  # 128KB em GB
    predictions = np.zeros_like(access_counts, dtype=int)
    
    # Cria perfil do usuário para obter os parâmetros de custo
    user_profile = UserProfile(cost_weight=cost_weight)
    
    for i, (accesses, volume) in enumerate(zip(access_counts, volumes)):
        # Ajusta volume para o mínimo faturável
        billable_volume = max(volume, MIN_BILLABLE_SIZE)
        billable_volume = billable_volume / (1024 ** 3)  # bytes para GB
        billable_volume = 1
        
        # Calcula custo e latência para cada classe
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
        
        # Escolhe a classe com menor custo balanceado
        predictions[i] = np.argmin(costs)
    
    return predictions

def calculate_age(df_access, window_start):
    """
    Calcula a idade de cada objeto em semanas, relativa ao window_start.
    Idade = window_start - first_week_with_access (se >0, else 0)
    """
    ages = {}
    for idx, row in df_access.iterrows():
        first_access = next((i for i, val in enumerate(row) if val > 0), None)
        if first_access is not None:
            age = max(0, window_start - first_access)
        else:
            age = 0  # Nunca acessado
        ages[idx] = age
    return pd.Series(ages)

def run_analysis(window, user_profile, df_access, df_volume, models_to_run, classifiers, output_dir, window_size):
    """
    Executa a análise para uma janela específica.
    
    Args:
        window (dict): Dicionário com dados de treino e teste
        user_profile (UserProfile): Perfil do usuário
        df_access (DataFrame): Dados de acesso
        df_volume (DataFrame): Dados de volume
        models_to_run (list): Lista de modelos a serem executados
        classifiers (dict): Dicionário com os classificadores
        output_dir (str): Diretório de saída
        window_size (int): Tamanho da janela
    
    Returns:
        dict: Resultados da análise
    """
    print("\nPreparando dados...")
    first_week = window['train'][0]
    train_data = extract_data(df_access, *window['train'])
    test_data = extract_data(df_access, *window['test'])
    label_train_data = extract_data(df_access, *window['label_train'])
    label_test_data = extract_data(df_access, *window['label_test'])

    # Primeiro filtra por volume
    train_data = filter_by_volume(train_data, df_volume, first_week)
    test_data = filter_by_volume(test_data, df_volume, first_week)
    label_train_data = filter_by_volume(label_train_data, df_volume, first_week)
    label_test_data = filter_by_volume(label_test_data, df_volume, first_week)

    # Depois calcula as idades apenas para os objetos já filtrados
    train_ages = calculate_age(df_access, window['train'][0])
    test_ages = calculate_age(df_access, window['test'][0])

    # Filtra as idades para corresponder aos índices dos dados já filtrados
    train_ages = train_ages[train_ages.index.isin(train_data.index)]
    test_ages = test_ages[test_ages.index.isin(test_data.index)]

    # Adiciona coluna age aos dados
    train_data['age'] = train_ages
    test_data['age'] = test_ages
    
    # Carrega dados de volume
    volumes = df_volume.loc[:, df_volume.columns[-1]]
    
    # Gera rótulos multiclasse
    y_train = get_label(label_train_data, volumes, user_profile.cost_weight)
    y_test = get_label(label_test_data, volumes, user_profile.cost_weight)

    # imprimindo quantos objetos temos em cada classe para depuração
    print(f"\nDistribuição das classes (treino):")
    print(y_train.value_counts())
    print(f"\nDistribuição das classes (teste):")
    print(y_test.value_counts())
    
    print(f"\nDistribuição das classes (treino):")
    print(y_train.value_counts(normalize=True))
    print(f"\nDistribuição das classes (teste):")
    print(y_test.value_counts(normalize=True))
    
    # Balanceia dados de treino
    print("\nBalanceando dados de treino...")
    
    # Verifica se há pelo menos 2 classes para aplicar SMOTE
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        print(f"AVISO: Apenas {len(unique_classes)} classe(s) encontrada(s). Pulando balanceamento.")
        X_train_bal, y_train_bal = train_data, y_train
    else:
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
    
    print("\nNormalizando dados...")
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
            print("Aplicando estratégia Always Hot...")
        elif model_name == 'AWL':
            y_pred = always_warm_predictions
            print("Aplicando estratégia Always Warm...")
        elif model_name == 'ACL':
            y_pred = always_cold_predictions
            print("Aplicando estratégia Always Cold...")
        elif model_name == 'ONL':
            print("Aplicando estratégia Online...")
            y_pred = get_online_predictions(test_data, volumes, user_profile.cost_weight)
        else:
            try:
                model = set_classifier(model_name, classifiers)
                print("Iniciando treinamento...")
                model.fit(X_train_scaled, y_train_bal)
                print("Treinamento concluído. Realizando predições...")
                
                if model_name == "HV":
                    y_pred = model.predict(X_test_scaled)
                else:
                    y_prob = model.predict_proba(X_test_scaled)
                    y_pred = np.zeros_like(y_test)
                    
                    # Aplica thresholds baseados no cost_weight
                    y_pred = apply_cost_weight_thresholds(y_prob, user_profile.cost_weight, user_profile)
                        
            except Exception as e:
                print(f"ERRO ao treinar/predizer com {model_name}: {str(e)}")
                continue
        
        print("\nCalculando métricas...")
        access_counts = label_test_data.sum(axis=1)
        volumes = test_data.index.map(lambda x: df_volume.loc[x, df_volume.columns[-1]])
        
        # Calcular métricas multiclasse
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
        
        print(f"\nResultados para {model_name}:")
        print(f"- Custo: {results[model_name]['cost']:,.2f}")
        print(f"- Latência: {results[model_name]['latency']:,.2f} ms")
        print(f"- Precisão: {precision:.4f}")
        print(f"- Recall: {recall:.4f}")
        print(f"- F1-Score: {f1:.4f}")
        print(f"- Matriz de Confusão:")
        print(cm)
    
    return results

def save_final_results(final_results, output_dir):
    """
    Salva os resultados finais em um arquivo CSV.

    Args:
        final_results (dict): Resultados consolidados.
        output_dir (str): Diretório de saída.
    """
    results_for_csv = []
    for profile_key, results in final_results.items():
        # Extrai model_name e cost_weight da chave (formato: "model_name_cost_weight")
        parts = profile_key.split('_')
        if len(parts) >= 2:
            # Pega o último elemento como cost_weight e o resto como model_name
            cost_weight = float(parts[-1])
            model_name = '_'.join(parts[:-1])
        else:
            # Fallback para formato antigo
            model_name, cost_weight = profile_key.split('_')
            cost_weight = float(cost_weight)
        cm = results['confusion_matrix']
        
        # Extrair métricas da matriz de confusão para cada classe
        cm_shape = cm.shape
        if cm_shape == (2, 2):
            # Matriz 2x2 - apenas 2 classes presentes
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
            # Matriz 3x3 - todas as 3 classes presentes
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
            # Formato inesperado - usar zeros
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
    output_csv_path = os.path.join(output_dir, 'resultados_finais.csv')
    results_df.to_csv(output_csv_path, index=False)
    print(f"Resultados finais salvos em CSV: {output_csv_path}")

def accumulate_results(final_results, window_results, cost_weight):
    """
    Acumula os resultados de uma janela nos resultados finais.

    Args:
        final_results (dict): Dicionário com os resultados acumulados.
        window_results (dict): Resultados da janela atual.
        cost_weight (float): Peso do custo usado na análise.
    """
    for model_name, result in window_results.items():
        profile_key = f"{model_name}_{cost_weight:.1f}"
        
        if profile_key not in final_results:
            # Inicializar acumuladores para o modelo
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
                'confusion_matrix': np.zeros((3, 3)),  # Matriz 3x3 para três classes
                'n_windows': 0
            }
        
        # Extrair y_true e y_pred da matriz de confusão
        cm = result['confusion_matrix']
        y_true = []
        y_pred = []
        
        # Verifica o tamanho da matriz de confusão
        cm_shape = cm.shape
        if cm_shape == (2, 2):
            # Matriz 2x2 - apenas 2 classes presentes
            for i in range(2):
                for j in range(2):
                    y_true.extend([i] * cm[i][j])
                    y_pred.extend([j] * cm[i][j])
        elif cm_shape == (3, 3):
            # Matriz 3x3 - todas as 3 classes presentes
            for i in range(3):
                for j in range(3):
                    y_true.extend([i] * cm[i][j])
                    y_pred.extend([j] * cm[i][j])
        else:
            print(f"AVISO: Matriz de confusão com formato inesperado: {cm_shape}")
            continue
        
        # Calcular métricas para a janela atual
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Atualizar acumuladores
        n = final_results[profile_key]['n_windows']
        final_results[profile_key]['model_cost'] += result['cost']
        final_results[profile_key]['oracle_cost'] += result['oracle_cost']
        final_results[profile_key]['model_latency'] += result['latency']
        final_results[profile_key]['oracle_latency'] += result['oracle_latency']
        
        # Soma a matriz de confusão considerando diferentes tamanhos
        cm_shape = cm.shape
        stored_cm = final_results[profile_key]['confusion_matrix']
        
        if cm_shape == (2, 2) and stored_cm.shape == (3, 3):
            # Expandir matriz 2x2 para 3x3
            expanded_cm = np.zeros((3, 3))
            expanded_cm[:2, :2] = cm
            final_results[profile_key]['confusion_matrix'] += expanded_cm
        elif cm_shape == (3, 3) and stored_cm.shape == (3, 3):
            # Ambas são 3x3
            final_results[profile_key]['confusion_matrix'] += cm
        elif cm_shape == (2, 2) and stored_cm.shape == (2, 2):
            # Ambas são 2x2
            final_results[profile_key]['confusion_matrix'] += cm
        else:
            print(f"AVISO: Incompatibilidade de matrizes de confusão: {cm_shape} vs {stored_cm.shape}")
        
        final_results[profile_key]['n_windows'] += 1
        
        # Atualizar métricas com média móvel
        final_results[profile_key]['accuracy'] = ((n * final_results[profile_key]['accuracy'] + acc) / (n + 1))
        final_results[profile_key]['precision'] = ((n * final_results[profile_key]['precision'] + prec) / (n + 1))
        final_results[profile_key]['recall'] = ((n * final_results[profile_key]['recall'] + rec) / (n + 1))
        final_results[profile_key]['f1'] = ((n * final_results[profile_key]['f1'] + f1) / (n + 1))

def load_configuration(config_path):
    """Carrega a configuração do framework."""
    print("=== Iniciando execução do framework ===")
    print("\nCarregando dados e parâmetros...")
    
    data_loader = DataLoader(data_dir='data/', config_path=config_path)
    params = data_loader.params
    
    print(f"Configurações carregadas:")
    print(f"- Arquivo de configuração: {config_path}")
    print(f"- População: {params['pop_name']}")
    print(f"- Tamanho da janela: {params['window_size']}")
    print(f"- Passo: {params['step_size']}")
    print(f"- Modelos: {params['models_to_run']}")
    
    return data_loader, params

def initialize_components(params):
    """Inicializa classificadores e carrega dados."""
    print("\nInicializando classificadores...")
    classifiers = get_default_classifiers()

    print("\nCarregando dados de acesso e volume...")
    data_loader = DataLoader(data_dir='data/', config_path='config/config.yaml')
    df_access = data_loader.load_access_data(params['pop_name'])
    df_volume = data_loader.load_volume_data(params['pop_name'])
    df_access.set_index('NameSpace', inplace=True)
    df_volume.set_index('NameSpace', inplace=True)
    
    print(f"Dimensões dos dados:")
    print(f"- Acessos: {df_access.shape}")
    print(f"- Volumes: {df_volume.shape}")
    
    return classifiers, df_access, df_volume

def setup_output_directories(pop_name, window_size, step_size):
    """Configura os diretórios de saída para as análises."""
    output_dir = os.path.join('results', f'results_{pop_name}_{window_size}_{step_size}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def process_window(window, window_index, total_windows, user_profile, df_access, df_volume, 
                  models_to_run, classifiers, output_dir, window_size):
    """Processa uma janela específica"""
    print(f"\n=== Processando janela {window_index+1}/{total_windows} ===")

    print("\n--- Executando análise ---")
    results = run_analysis(window, user_profile, df_access, df_volume, 
                                   models_to_run, classifiers, output_dir, 
                                   window_size)
    
    return results

def setup_experiment_parameters(params):
    """Configura os parâmetros do experimento."""
    window_size = params['window_size']
    step_size = params['step_size']
    models_to_run = params['models_to_run']
    cost_weights = params.get('cost_weight', 0.5)
    
    # Garante que cost_weights seja uma lista
    if not isinstance(cost_weights, list):
        cost_weights = [cost_weights]
    
    return window_size, step_size, models_to_run, cost_weights

def generate_time_windows(df_access, window_size, step_size):
    """Gera as janelas temporais para o experimento."""
    windows = get_time_windows(df_access, window_size, step_size)
    print(f"Geradas {len(windows)} janelas temporais para processamento")
    return windows

def initialize_result_containers():
    """Inicializa os containers para armazenar os resultados."""
    final_results = {}
    return final_results

def accumulate_window_results(final_results,  
                            results, cost_weight):
    """Acumula os resultados de uma janela nos containers finais."""
    accumulate_results(final_results, results, cost_weight)

def process_all_windows(df_access, df_volume, params, classifiers, output_dir):
    """Processa todas as janelas temporais."""
    print("\nIniciando processamento das janelas...")
    
    # Configura parâmetros do experimento
    window_size, step_size, models_to_run, cost_weights = setup_experiment_parameters(params)
    
    # Gera janelas temporais
    windows = generate_time_windows(df_access, window_size, step_size)
    
    # Inicializa containers de resultados
    final_results = initialize_result_containers()
    
    # Processa cada valor de cost_weight
    for cost_weight in cost_weights:
        print(f"\n=== Processando com cost_weight = {cost_weight} ===")
        user_profile = UserProfile(cost_weight)
        
        # Processa cada janela para este cost_weight
        for i, window in enumerate(windows):
            results = process_window(
                window, i, len(windows), user_profile, df_access, df_volume,
                models_to_run, classifiers, output_dir, window_size
            )
            
            # Acumula resultados
            accumulate_window_results(
                final_results,
                results, cost_weight
            )
    
    return final_results, len(windows) * len(cost_weights)

def save_results(final_results, output_dir, total_windows):
    """Salva os resultados finais."""
    save_final_results(final_results, output_dir)
    print(f"Total de janelas processadas: {total_windows}")

def main():
    """Função principal que executa o framework de classificação multiclasse."""
    # Configuração do parser de argumentos
    parser = argparse.ArgumentParser(description='Framework de classificação multiclasse para armazenamento')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Caminho para o arquivo de configuração (padrão: config/config.yaml)')
    
    args = parser.parse_args()
    
    # Carrega configuração
    data_loader, params = load_configuration(args.config)
    
    # Inicializa componentes
    classifiers, df_access, df_volume = initialize_components(params)
    
    # Configura diretórios de saída
    output_dir = setup_output_directories(
        params['pop_name'], params['window_size'], params['step_size']
    )
    
    # Processa todas as janelas
    final_results, total_windows = process_all_windows(
        df_access, df_volume, params, classifiers, output_dir
    )
    
    # Salva resultados
    save_results(final_results, output_dir, total_windows)

if __name__ == '__main__':
    main()
