import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def accumulate_results(final_results, window_results, cost_weight):
    """
    Acumula os resultados de uma janela nos resultados finais.

    Args:
        final_results (dict): Dicionário com os resultados acumulados.
        window_results (dict): Resultados da janela atual.
        cost_weight (float): Peso do custo usado na análise.
    """
    for model_name, result in window_results.items():
        profile_key = f"{model_name}_{int(cost_weight)}"
        
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
        for i in range(3):  # Para cada classe
            for j in range(3):
                y_true.extend([i] * cm[i][j])
                y_pred.extend([j] * cm[i][j])
        
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
        final_results[profile_key]['confusion_matrix'] += cm
        final_results[profile_key]['n_windows'] += 1
        
        # Atualizar métricas com média móvel
        final_results[profile_key]['accuracy'] = ((n * final_results[profile_key]['accuracy'] + acc) / (n + 1))
        final_results[profile_key]['precision'] = ((n * final_results[profile_key]['precision'] + prec) / (n + 1))
        final_results[profile_key]['recall'] = ((n * final_results[profile_key]['recall'] + rec) / (n + 1))
        final_results[profile_key]['f1'] = ((n * final_results[profile_key]['f1'] + f1) / (n + 1))