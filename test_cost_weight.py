#!/usr/bin/env python3
"""
Script de teste para demonstrar a funcionalidade de cost_weight.
Este script mostra como os thresholds são ajustados baseado no cost_weight.
"""

import numpy as np
from models.user_profiles import UserProfile
from main import apply_cost_weight_thresholds

def test_cost_weight_thresholds():
    """Testa a aplicação de thresholds baseados no cost_weight."""
    
    print("=== Teste da Funcionalidade Cost Weight ===\n")
    
    # Simula probabilidades de um modelo (3 classes: COLD, WARM, HOT)
    # Cada linha representa uma amostra com probabilidades [COLD, WARM, HOT]
    y_prob = np.array([
        [0.3, 0.4, 0.3],  # Amostra 1: WARM mais provável
        [0.1, 0.2, 0.7],  # Amostra 2: HOT mais provável
        [0.6, 0.3, 0.1],  # Amostra 3: COLD mais provável
        [0.2, 0.3, 0.5],  # Amostra 4: HOT mais provável
        [0.4, 0.5, 0.1],  # Amostra 5: WARM mais provável
    ])
    
    print("Probabilidades originais:")
    print("Amostra | COLD  | WARM  | HOT   | Classe Original")
    print("-" * 50)
    for i, probs in enumerate(y_prob):
        original_class = np.argmax(probs)
        print(f"   {i+1}   | {probs[0]:.3f} | {probs[1]:.3f} | {probs[2]:.3f} | {original_class}")
    
    print("\n" + "="*60)
    
    # Testa diferentes valores de cost_weight
    cost_weights = [0.2, 0.5, 0.8]
    
    for cost_weight in cost_weights:
        print(f"\n--- Testando cost_weight = {cost_weight:.1f} ---")
        
        # Cria perfil do usuário
        user_profile = UserProfile(cost_weight=cost_weight)
        
        # Aplica thresholds
        predictions = apply_cost_weight_thresholds(y_prob, cost_weight, user_profile)
        
        # Mostra resultados
        print("Amostra | COLD  | WARM  | HOT   | Classe Original | Classe Ajustada | Mudança")
        print("-" * 75)
        for i, (probs, pred) in enumerate(zip(y_prob, predictions)):
            original_class = np.argmax(probs)
            change = "✓" if original_class == pred else "✗"
            print(f"   {i+1}   | {probs[0]:.3f} | {probs[1]:.3f} | {probs[2]:.3f} |       {original_class}       |       {pred}        |   {change}")
        
        # Estatísticas
        original_predictions = np.argmax(y_prob, axis=1)
        changes = np.sum(original_predictions != predictions)
        print(f"\nMudanças: {changes}/{len(predictions)} amostras ({changes/len(predictions)*100:.1f}%)")
        
        # Distribuição das classes
        print("Distribuição das classes:")
        print(f"- Original: COLD={np.sum(original_predictions==0)}, WARM={np.sum(original_predictions==1)}, HOT={np.sum(original_predictions==2)}")
        print(f"- Ajustada: COLD={np.sum(predictions==0)}, WARM={np.sum(predictions==1)}, HOT={np.sum(predictions==2)}")

def test_threshold_calculation():
    """Testa o cálculo dos thresholds para diferentes valores de cost_weight."""
    
    print("\n=== Cálculo dos Thresholds ===\n")
    
    print("cost_weight | Threshold HOT | Threshold WARM | Interpretação")
    print("-" * 65)
    
    for cost_weight in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # Calcula thresholds (mesma lógica da função apply_cost_weight_thresholds)
        base_hot_threshold = 0.4
        hot_threshold = base_hot_threshold + (cost_weight - 0.5) * 0.8
        warm_threshold = 0.3 + (cost_weight - 0.5) * 0.4
        
        # Interpretação
        if cost_weight < 0.4:
            interpretation = "Muito Agressivo"
        elif cost_weight < 0.6:
            interpretation = "Neutro"
        else:
            interpretation = "Conservador"
        
        print(f"    {cost_weight:.1f}    |     {hot_threshold:.3f}     |     {warm_threshold:.3f}     | {interpretation}")

if __name__ == "__main__":
    test_cost_weight_thresholds()
    test_threshold_calculation()
    
    print("\n=== Resumo ===\n")
    print("• cost_weight baixo (0.1-0.3): Favorece latência, mais objetos classificados como HOT")
    print("• cost_weight médio (0.4-0.6): Balanceado entre custo e latência")
    print("• cost_weight alto (0.7-0.9): Favorece custo, menos objetos classificados como HOT")
    print("\nPara usar no framework principal, configure o cost_weight no arquivo config/config.yaml") 