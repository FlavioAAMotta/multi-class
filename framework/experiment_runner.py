import os
from typing import Dict, List, Any
import pandas as pd
from data_loader import DataLoader
from models.classifiers import get_default_classifiers
from utils.time_window import get_time_windows
from utils.metrics import accumulate_results
from framework.analysis import run_analysis

class ExperimentRunner:
    """
    Classe responsável por executar experimentos de classificação multi-classe
    com diferentes modelos e configurações.
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Inicializa o experiment runner.
        
        Args:
            config_path: Caminho para o arquivo de configuração
        """
        self.data_loader = DataLoader(data_dir='data/', config_path=config_path)
        self.params = self.data_loader.params
        self.classifiers = None
        self.df_access = None
        self.df_volume = None
        self.final_results = {}
        
    def initialize(self):
        """Inicializa classificadores e carrega dados."""
        print("=== Iniciando execução do framework ===")
        
        print("\nInicializando classificadores...")
        self.classifiers = get_default_classifiers()
        
        print("\nCarregando dados de acesso e volume...")
        pop_name = self.params['pop_name']
        self.df_access = self.data_loader.load_access_data(pop_name)
        self.df_volume = self.data_loader.load_volume_data(pop_name)
        self.df_access.set_index('NameSpace', inplace=True)
        self.df_volume.set_index('NameSpace', inplace=True)
        
    def setup_output_directory(self) -> str:
        """Configura o diretório de saída para os resultados."""
        window_size = self.params['window_size']
        step_size = self.params['step_size']
        pop_name = self.params['pop_name']
        
        output_dir = os.path.join('results', f'results_{pop_name}_{window_size}_{step_size}')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
        
    def run_experiment(self):
        """Executa o experimento completo."""
        self.initialize()
        output_dir = self.setup_output_directory()
        
        window_size = self.params['window_size']
        step_size = self.params['step_size']
        models_to_run = self.params['models_to_run']
        cost_weight = self.params.get('cost_weight', 50)
        
        windows = get_time_windows(self.df_access, window_size, step_size)
        
        print(f"\nProcessando {len(windows)} janelas temporais...")
        
        for i, window in enumerate(windows):
            print(f"Processando janela {i+1}/{len(windows)}")
            results = run_analysis(
                window, self.df_access, self.df_volume,
                models_to_run, self.classifiers, output_dir, window_size
            )
            accumulate_results(self.final_results, results, cost_weight)
            
        return self.final_results
    
    def save_results(self, results: Dict[str, Any], output_path: str = None):
        """Salva os resultados do experimento."""
        if output_path is None:
            output_path = os.path.join('results', 'final_results.json')
            
        # Implementar salvamento dos resultados
        print(f"Resultados salvos em: {output_path}")
        
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Retorna um resumo do experimento executado."""
        return {
            'pop_name': self.params['pop_name'],
            'window_size': self.params['window_size'],
            'step_size': self.params['step_size'],
            'models_run': self.params['models_to_run'],
            'total_windows': len(get_time_windows(self.df_access, 
                                                self.params['window_size'], 
                                                self.params['step_size'])),
            'results_keys': list(self.final_results.keys())
        } 