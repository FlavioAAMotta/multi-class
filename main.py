from framework.experiment_runner import ExperimentRunner
import argparse
import sys

def main():
    """
    Função principal que executa o framework de classificação multi-classe.
    """
    parser = argparse.ArgumentParser(description='Framework de Classificação Multi-Classe')
    parser.add_argument('--config', '-c', 
                       default='config/config.yaml',
                       help='Caminho para o arquivo de configuração')
    parser.add_argument('--output', '-o',
                       help='Caminho para salvar os resultados')
    
    args = parser.parse_args()
    
    try:
        # Criar e executar o experimento
        runner = ExperimentRunner(config_path=args.config)
        results = runner.run_experiment()
        
        # Salvar resultados
        runner.save_results(results, args.output)
        
        # Exibir resumo
        summary = runner.get_experiment_summary()
        print("\n=== Resumo do Experiment ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
