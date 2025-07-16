import numpy as np
from .user_profiles import UserProfile

def get_online_predictions(test_data, volumes, cost_weight=0.5):
    """
    Classifica dados de teste baseado no conhecimento dos custos e número de acessos.
    
    Args:
        test_data: Dados de teste contendo número de acessos
        volumes: Volumes dos dados
        cost_weight: Peso do custo para balanceamento (0-1)
    
    Returns:
        predictions: Array com classificações (0=COLD, 1=WARM, 2=HOT)
    """
    # Cria perfil do usuário com os pesos especificados
    user_profile = UserProfile(cost_weight=cost_weight)
    
    # Extrai número de acessos dos dados de teste
    # Calcula o total de acessos por objeto (soma das colunas)
    if hasattr(test_data, 'sum'):
        # Se for DataFrame, soma as colunas para obter total de acessos por objeto
        acessos = test_data.sum(axis=1).values
    else:
        # Se for array, assume que já é o número de acessos por objeto
        acessos = np.array(test_data)
    
    # Volume fixo para cálculos (usando média dos volumes se disponível)
    if volumes is not None:
        volume_fixo = np.mean(volumes) if len(volumes) > 0 else 1.0
    else:
        volume_fixo = 1.0
    
    predictions = []
    
    # Para cada item nos dados de teste
    for i, num_acessos in enumerate(acessos):
        # Calcula custos para as três classes de armazenamento
        custo_hot = (user_profile.hot_storage_cost * volume_fixo) + \
                   (user_profile.hot_operation_cost * num_acessos * volume_fixo) + \
                   (user_profile.hot_retrieval_cost * num_acessos * volume_fixo)
        
        custo_warm = (user_profile.warm_storage_cost * volume_fixo) + \
                    (user_profile.warm_operation_cost * num_acessos * volume_fixo) + \
                    (user_profile.warm_retrieval_cost * num_acessos * volume_fixo)
        
        custo_cold = (user_profile.cold_storage_cost * volume_fixo) + \
                    (user_profile.cold_operation_cost * num_acessos * volume_fixo) + \
                    (user_profile.cold_retrieval_cost * num_acessos * volume_fixo)
        
        # Classifica baseado no menor custo
        custos = [custo_cold, custo_warm, custo_hot]
        classe_otima = np.argmin(custos)
        
        # Aplica threshold de decisão baseado no perfil do usuário
        # Se o custo_weight for alto (conservador), tende a classificar como COLD/WARM
        # Se o custo_weight for baixo (agressivo), tende a classificar como HOT
        if user_profile.cost_weight > 0.7:  # Conservador
            # Penaliza HOT para usuários conservadores
            custos[2] *= 1.2  # Aumenta custo do HOT em 20%
            classe_otima = np.argmin(custos)
        elif user_profile.cost_weight < 0.3:  # Agressivo
            # Favorece HOT para usuários agressivos
            custos[2] *= 0.8  # Reduz custo do HOT em 20%
            classe_otima = np.argmin(custos)
        
        predictions.append(classe_otima)
    
    return np.array(predictions)

def get_online_predictions_with_thresholds(test_data, volumes, cost_weight, use_thresholds=True):
    """
    Versão alternativa que também considera thresholds de decisão baseados no perfil.
    
    Args:
        test_data: Dados de teste contendo número de acessos
        volumes: Volumes dos dados
        cost_weight: Peso do custo para balanceamento (0-1)
        use_thresholds: Se deve usar thresholds de decisão
    
    Returns:
        predictions: Array com classificações (0=COLD, 1=WARM, 2=HOT)
    """
    user_profile = UserProfile(cost_weight=cost_weight)
    
    # Calcula o total de acessos por objeto (soma das colunas)
    if hasattr(test_data, 'sum'):
        # Se for DataFrame, soma as colunas para obter total de acessos por objeto
        acessos = test_data.sum(axis=1).values
    else:
        # Se for array, assume que já é o número de acessos por objeto
        acessos = np.array(test_data)
    
    if volumes is not None:
        volume_fixo = np.mean(volumes) if len(volumes) > 0 else 1.0
    else:
        volume_fixo = 1.0
    
    predictions = []
    
    for num_acessos in acessos:
        if use_thresholds:
            # Usa thresholds baseados no perfil do usuário
            # Calcula um score baseado no número de acessos e volume
            score = (num_acessos * volume_fixo) / 1000.0  # Normaliza o score
            
            # Ajusta threshold baseado no perfil do usuário
            adjusted_cold_threshold = user_profile.cold_threshold * (1 + user_profile.cost_weight * 0.5)
            adjusted_warm_threshold = user_profile.warm_threshold * (1 + user_profile.cost_weight * 0.3)
            
            if score <= adjusted_cold_threshold:
                classe = 0  # COLD
            elif score <= adjusted_warm_threshold:
                classe = 1  # WARM
            else:
                classe = 2  # HOT
        else:
            # Usa apenas cálculo de custos
            custo_hot = (user_profile.hot_storage_cost * volume_fixo) + \
                       (user_profile.hot_operation_cost * num_acessos * volume_fixo) + \
                       (user_profile.hot_retrieval_cost * num_acessos * volume_fixo)
            
            custo_warm = (user_profile.warm_storage_cost * volume_fixo) + \
                        (user_profile.warm_operation_cost * num_acessos * volume_fixo) + \
                        (user_profile.warm_retrieval_cost * num_acessos * volume_fixo)
            
            custo_cold = (user_profile.cold_storage_cost * volume_fixo) + \
                        (user_profile.cold_operation_cost * num_acessos * volume_fixo) + \
                        (user_profile.cold_retrieval_cost * num_acessos * volume_fixo)
            
            classe = np.argmin([custo_cold, custo_warm, custo_hot])
        
        predictions.append(classe)
    
    return np.array(predictions)
    