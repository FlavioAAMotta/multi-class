from enum import Enum
from dataclasses import dataclass

class ProfileType(Enum):
    PROFILE_A = "A"  # 80% Price, 20% Latency
    PROFILE_B = "B"  # 20% Price, 80% Latency
    PROFILE_C = "C"  # 50% Price, 50% Latency

@dataclass
class UserProfile:
    """
    User profile class that handles risk preference between cost and latency.
    Higher cost_weight means more conservative in predicting HOT (prefers lower costs)
    Higher latency_weight means more aggressive in predicting HOT (prefers lower latency)
    """
    def __init__(self, cost_weight=0.5):
        """
        Inicializa o perfil do usuário.
        
        Args:
            cost_weight (float): Peso do custo no balanceamento entre custo e latência (0-1)
        """
        self.cost_weight = cost_weight

        # Parâmetros de custo
        self.hot_storage_cost = 0.0400   # Custo de armazenamento HOT
        self.warm_storage_cost = 0.0150  # Custo de armazenamento WARM
        self.cold_storage_cost = 0.0040  # Custo de armazenamento COLD
        
        self.hot_operation_cost = 0.0004   # Custo de operação HOT
        self.warm_operation_cost = 0.0010  # Custo de operação WARM
        self.cold_operation_cost = 0.0020  # Custo de operação COLD
        
        self.hot_retrieval_cost = 0.0060   # Custo de recuperação HOT
        self.warm_retrieval_cost = 0.0110  # Custo de recuperação WARM
        self.cold_retrieval_cost = 0.0200  # Custo de recuperação COLD
        
        # Parâmetros de latência (ms)
        self.hot_latency = 400.0    # Latência de acesso HOT
        self.warm_latency = 800.0   # Latência de acesso WARM
        self.cold_latency = 1200.0  # Latência de acesso COLD
        
        # Thresholds de decisão para cada classe
        self.hot_threshold = 1.0
        self.warm_threshold = 0.66
        self.cold_threshold = 0.33
        
        self.latency_weight = (100 - cost_weight) / 100.0
        
        # Calculate decision threshold based on weights
        # Higher cost_weight = higher threshold = more conservative in predicting HOT
        self.decision_threshold = 0.5 + (self.cost_weight - 0.5) * 0.4
        
    def __str__(self):
        return f"UserProfile(cost_weight={self.cost_weight:.2f}, latency_weight={self.latency_weight:.2f}, decision_threshold={self.decision_threshold:.2f})" 