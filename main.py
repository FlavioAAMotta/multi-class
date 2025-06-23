from data_loader import DataLoader
from models.classifiers import get_default_classifiers, set_classifier
from models.onl import get_online_predictions
from utils.time_window import get_time_windows
from utils.metrics import accumulate_results

def main():
    print("=== Iniciando execução do framework ===")

    data_loader = DataLoader(data_dir='data/', config_path='config/config.yaml')
    params = data_loader.params
    window_size = params['window_size']
    step_size = params['step_size']
    pop_name = params['pop_name']
    models_to_run = params['models_to_run']

    profile_config = params.get('profile', {})
    cost_weight = profile_config.get('cost_weight', 50)

    print("\nInicializando classificadores...")
    classifiers = get_default_classifiers()
    
    print("\nCarregando dados de acesso e volume...")
    df_access = data_loader.load_access_data(pop_name)
    df_volume = data_loader.load_volume_data(pop_name)
    df_access.set_index('NameSpace', inplace=True)
    df_volume.set_index('NameSpace', inplace=True)
    
    windows = get_time_windows(df_access, window_size, step_size)

    final_results = {}

    for i, window in enumerate(windows):
        results = run_analysis(x_train, y_train, x_test, y_test, classifiers, models_to_run)
        accumulate_results(final_results, results, cost_weight)

        

if __name__ == "__main__":
    main()
