import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metric_trends(pop_name):
    """
    Plota as tendências das métricas para uma população específica.
    
    Args:
        pop_name (str): Nome da população (ex: 'Pop1', 'Pop2')
    """
    # Configurar diretórios
    input_dir = f'analysis_results/sensitivity/{pop_name}/4x4'
    output_dir = 'analysis_results/trends'
    os.makedirs(output_dir, exist_ok=True)
    
    # Ler os resultados
    df = pd.read_csv(os.path.join(input_dir, 'all_weights_results.csv'))
    
    # Definir métricas para plotar
    metrics = {
        'total_model_cost': 'Total Cost',
        'total_model_latency': 'Total Latency (s)',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1-Score'
    }
    
    # Configurar estilo personalizado
    plt.style.use('default')
    
    # Definir cores e marcadores para cada modelo
    model_styles = {
        'SVML': {'marker': '^', 'color': '#0000FF'},    # Azul
        'SV-Grid': {'marker': 'D', 'color': '#FF00FF'}, # Magenta
        'SV': {'marker': '*', 'color': '#FFD700'},      # Dourado
        'SVMR': {'marker': 'v', 'color': '#FF8C00'},    # Laranja escuro
        'LR': {'marker': 'h', 'color': '#00FFFF'},      # Ciano
        'RF': {'marker': '8', 'color': '#FF1493'},      # Rosa escuro
        'DCT': {'marker': '<', 'color': '#32CD32'},     # Verde limão
        'KNN': {'marker': '>', 'color': '#4B0082'},     # Índigo
        'HV': {'marker': '+', 'color': '#FF4500'}       # Laranja avermelhado
    }
    
    # Plotar cada métrica
    for metric, metric_label in metrics.items():
        plt.figure(figsize=(12, 6))
        
        for model in df['model'].unique():
            if model.startswith('AWL') or model.startswith('ONL') or model.startswith('AHL'):
                continue
            model_base = model.split('_')[0]
            model_data = df[df['model'] == model]
            
            if model_base in model_styles:
                style = model_styles[model_base]
                plt.plot(model_data['cost_weight'],
                        model_data[metric],
                        marker=style['marker'],
                        color=style['color'],
                        label=model,
                        markersize=8,
                        linewidth=2,
                        alpha=0.8)
        
        plt.xlabel('Cost Weight', fontsize=20)
        plt.ylabel(metric_label, fontsize=20)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1),
                  loc='upper left',
                  borderaxespad=0.,
                  fontsize=20,
                  frameon=True,
                  shadow=True)
        
        # Ajustar layout e salvar
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{pop_name}_{metric}_trend.svg'),
                    format='svg',
                    bbox_inches='tight',
                    dpi=300)
        plt.savefig(os.path.join(output_dir, f'{pop_name}_{metric}_trend.png'),
                    format='png',
                    bbox_inches='tight',
                    dpi=300)
        plt.close()

if __name__ == "__main__":
    # Plotar para ambas as populações
    plot_metric_trends("Pop1")
    plot_metric_trends("Pop2")
    print("Gráficos de tendência salvos em 'analysis_results/trends'") 