def get_time_windows(data, window_size, step_size):
    """
    Gera índices de janelas temporais para treinamento e teste.
    
    A lógica é:
    - train: dados de treino (período inicial)
    - label_train: rótulos para os dados de treino (período seguinte)
    - test: dados de teste (mesmo período dos rótulos de treino)
    - label_test: rótulos para os dados de teste (período seguinte)

    Args:
        data (DataFrame): Dados com colunas representando semanas.
        window_size (int): Número de semanas para cada janela.
        step_size (int): Intervalo de deslocamento entre janelas.

    Returns:
        list: Lista de dicionários com os índices de cada janela.
    """
    windows = []
    num_weeks = len(data.columns)
    
    # Para cada janela, precisamos de 3 períodos: treino, rótulos treino/teste, rótulos teste
    # total_window_size = window_size * 3
    total_window_size = window_size * 3
    
    for start in range(0, num_weeks - total_window_size + 1, step_size):
        windows.append({
            'train': (start, start + window_size),                    # Dados de treino
            'label_train': (start + window_size, start + 2 * window_size),  # Rótulos para treino
            'test': (start + window_size, start + 2 * window_size),   # Dados de teste (mesmo período dos rótulos de treino)
            'label_test': (start + 2 * window_size, start + 3 * window_size) # Rótulos para teste
        })
    return windows