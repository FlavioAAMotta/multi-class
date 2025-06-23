
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
            'test': (start + 2 * window_size, start + 3 * window_size),
            'label_test': (start + 3 * window_size, start + 4 * window_size)
        })
    return windows