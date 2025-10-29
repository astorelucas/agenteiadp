"""
AutoGluon Time Series Forecasting
"""

from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


def load_and_prepare_data(
    file_path: str,
    timestamp_column: str,
    item_id_value: str = None,
    freq: str = None,
    fill_missing: bool = True,
    fill_method: str = "auto"
):
    """
    Carrega e prepara dados de série temporal.
    
    Args:
        file_path: Caminho para o arquivo CSV
        timestamp_column: Nome da coluna de timestamp
        target_column: Nome da coluna alvo (padrão: "target")
        item_id_value: Valor para item_id (se None, usa nome do arquivo)
        freq: Frequência da série temporal (ex: 'H' para hora, 'D' para dia)
        fill_missing: Se True, preenche valores faltantes
        fill_method: Método para preencher valores faltantes ('auto', 'constant', 'forward')
    
    Returns:
        TimeSeriesDataFrame preparado
    """
    print(f"Carregando dados de: {file_path}")
    df = pd.read_csv(file_path)
    
    # Define item_id se não fornecido
    if item_id_value is None:
        item_id_value = Path(file_path).stem
    
    df['item_id'] = item_id_value
    
    # Cria TimeSeriesDataFrame
    data = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column='item_id',
        timestamp_column=timestamp_column
    )
    
    # Converte frequência se necessário
    if freq and data.freq != freq:
        print(f"Convertendo frequência para: {freq}")
        data = data.convert_frequency(freq=freq)
    
    # Preenche valores faltantes
    if fill_missing and data.isna().any().any():
        print(f"Preenchendo valores faltantes usando método: {fill_method}")
        data = data.fill_missing_values(method=fill_method)
    
    print(f"Dados carregados: {len(data)} registros, {len(data.item_ids)} séries temporais")
    print(f"Frequência inferida: {data.freq}")
    print(f"Comprimento médio das séries: {len(data) / len(data.item_ids):.0f}")
    
    return data


def train_forecast_model(
    train_data: TimeSeriesDataFrame,
    prediction_length: int,
    target_column: str = "target",
    eval_metric: str = "MASE",
    presets: str = "medium_quality",
    time_limit: int = None,
    num_val_windows: int = 1,
    quantile_levels: list = None,
    enable_ensemble: bool = True,
    hyperparameters: dict = None,
    model_path: str = "AutogluonModels",
    verbosity: int = 2
):
    """
    Treina modelo de previsão com AutoGluon.
    
    Args:
        train_data: Dados de treino
        prediction_length: Número de steps para prever
        target_column: Nome da coluna alvo
        eval_metric: Métrica de avaliação (MASE, WQL, MAE, RMSE, etc.)
        presets: Qualidade do modelo ('fast_training', 'medium_quality', 'high_quality', 'best_quality')
        time_limit: Tempo limite de treino em segundos
        num_val_windows: Número de janelas de validação (aumenta robustez)
        quantile_levels: Níveis de quantis para previsão probabilística
        enable_ensemble: Se True, cria ensemble dos modelos
        hyperparameters: Configuração manual de hiperparâmetros
        model_path: Caminho para salvar modelos
        verbosity: Nível de verbosidade (0-4)
    
    Returns:
        TimeSeriesPredictor treinado
    """
    print("\n" + "="*70)
    print("CONFIGURAÇÃO DO MODELO")
    print("="*70)
    print(f"Prediction Length: {prediction_length}")
    print(f"Evaluation Metric: {eval_metric}")
    print(f"Presets: {presets}")
    print(f"Time Limit: {time_limit}s" if time_limit else "Time Limit: None (treina até completar)")
    print(f"Validation Windows: {num_val_windows}")
    print(f"Enable Ensemble: {enable_ensemble}")
    print(f"Model Path: {model_path}")
    
    # Define quantile levels padrão se não fornecido
    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Cria predictor
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        target=target_column,
        eval_metric=eval_metric,
        path=model_path,
        quantile_levels=quantile_levels,
        verbosity=verbosity
    )
    
    # Argumentos de fit
    fit_kwargs = {
        'presets': presets,
        'num_val_windows': num_val_windows,
        'enable_ensemble': enable_ensemble
    }
    
    if time_limit:
        fit_kwargs['time_limit'] = time_limit
    
    if hyperparameters:
        fit_kwargs['hyperparameters'] = hyperparameters
    
    # Treina modelo
    print("\n" + "="*70)
    print("INICIANDO TREINAMENTO")
    print("="*70)
    predictor.fit(train_data, **fit_kwargs)
    
    return predictor


def evaluate_model(
    predictor: TimeSeriesPredictor,
    test_data: TimeSeriesDataFrame,
):
    """
    Avalia modelo.
    
    Args:
        predictor: Modelo treinado
        test_data: Dados de teste
    Returns:
        Dict com scores de avaliação
    """
    print("\n" + "="*70)
    print("AVALIAÇÃO DO MODELO")
    print("="*70)
    
    # Avaliação simples
    scores = predictor.evaluate(test_data)
    print(f"\nScores: {scores}")
    return scores



def analyze_models(
    predictor: TimeSeriesPredictor,
    test_data: TimeSeriesDataFrame = None
):
    """
    Analisa e compara modelos treinados.
    
    Args:
        predictor: Modelo treinado
        test_data: Dados de teste (opcional)
    """
    print("\n" + "="*70)
    print("ANÁLISE DE MODELOS")
    print("="*70)
    
    # Leaderboard
    if test_data is not None:
        print("\nLeaderboard (com dados de teste):")
        leaderboard = predictor.leaderboard(test_data)
    else:
        print("\nLeaderboard (validação interna):")
        leaderboard = predictor.leaderboard()
    
    print(leaderboard.to_string())
    
    # Informações adicionais
    print(f"\n{'='*70}")
    print("INFORMAÇÕES ADICIONAIS")
    print(f"{'='*70}")
    print(f"Melhor modelo: {predictor.model_best}")
    print(f"\nModelos treinados: {predictor.model_names()}")
    
    return leaderboard


def visualize_predictions(
    predictor: TimeSeriesPredictor,
    data: TimeSeriesDataFrame,
    predictions: TimeSeriesDataFrame = None,
    num_items: int = 4,
    max_history_length: int = 200,
    quantile_levels: list = None,
    save_path: str = None
):
    """
    Visualiza previsões do modelo.
    
    Args:
        predictor: Modelo treinado
        data: Dados (com histórico e forecast horizon)
        predictions: Previsões (se None, gera automaticamente)
        num_items: Número de séries temporais para plotar
        max_history_length: Comprimento máximo do histórico a mostrar
        quantile_levels: Quantis para visualizar
        save_path: Caminho para salvar figura
    """
    print("\n" + "="*70)
    print("VISUALIZANDO PREVISÕES")
    print("="*70)
    
    if predictions is None:
        print("Gerando previsões...")
        predictions = predictor.predict(data)
    
    if quantile_levels is None:
        quantile_levels = [0.1, 0.9]
    
    # Plot usando método nativo do AutoGluon
    predictor.plot(
        data=data,
        predictions=predictions,
        quantile_levels=quantile_levels,
        max_history_length=max_history_length,
        max_num_item_ids=num_items
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figura salva em: {save_path}")


def run_complete_forecast(
    file_path: str,
    timestamp_column: str,
    target_column: str,
    prediction_length: int,
    test_size_ratio: float = 0.2,
    eval_metric: str = "MASE",
    presets: str = "medium_quality",
    time_limit: int = None,
    num_val_windows: int = 1,
    freq: str = None,
    model_path: str = "AutogluonModels"
):
    """
    Executa pipeline completo de previsão de séries temporais.
    
    Args:
        file_path: Caminho para arquivo CSV
        timestamp_column: Nome da coluna de timestamp
        target_column: Nome da coluna alvo
        prediction_length: Número de steps para prever
        test_size_ratio: Proporção dos dados para teste
        eval_metric: Métrica de avaliação
        presets: Qualidade do modelo
        time_limit: Tempo limite de treino
        num_val_windows: Número de janelas de validação
        freq: Frequência da série temporal
        model_path: Caminho para salvar modelos
    
    Returns:
        Dict com predictor, scores e leaderboard
    """
    # 1. Carrega e prepara dados
    data = load_and_prepare_data(
        file_path=file_path,
        timestamp_column=timestamp_column,
        freq=freq
    )
    
    # 2. Divide em treino e teste
    print("\n" + "="*70)
    print("DIVISÃO TREINO/TESTE")
    print("="*70)
    train_data, test_data = data.train_test_split(prediction_length=prediction_length)
    print(f"Train data: {len(train_data)} registros")
    print(f"Test data: {len(test_data)} registros")
    
    # 3. Treina modelo
    predictor = train_forecast_model(
        train_data=train_data,
        prediction_length=prediction_length,
        target_column=target_column,
        eval_metric=eval_metric,
        presets=presets,
        time_limit=time_limit,
        num_val_windows=num_val_windows,
        model_path=model_path
    )
    
    # 4. Avalia modelo
    scores = evaluate_model(
        predictor=predictor,
        test_data=test_data
    )
    
    # 5. Analisa modelos
    leaderboard = analyze_models(
        predictor=predictor,
        test_data=test_data
    )
    
    predictions = predictor.predict(test_data)
    visualize_predictions(
        predictor=predictor,
        data=test_data,
        predictions=predictions,
        save_path=f"{model_path}/predictions_plot.png"
    )
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETO!")
    print("="*70)
    
    return {
        'predictor': predictor,
        'scores': scores,
        'leaderboard': leaderboard,
        'train_data': train_data,
        'test_data': test_data,
        'predictions': predictions
    }

def save_predictions_and_actuals(config, results):
    # Extrai informações necessárias
    model_path = config['model_path']

    predictions = results['predictions']
    test_data = results['test_data']

    target_col = config['target_column']
    prediction_length = config['prediction_length']

    # Extrai valores previstos e reais
    predicted_values = predictions['mean'].values
    real_values = test_data[target_col].iloc[-prediction_length:].values

    # Gera timestamps correspondentes
    timestamps = predictions.index.get_level_values('timestamp')
    timestamps_str = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps]

    # Salva previsões e valores reais em um csv
    df = pd.DataFrame({
        'timestamp': timestamps_str,
        'predictions': predicted_values,
        'real_values': real_values
    })

    output_path = f"{model_path}/predictions_and_actuals.csv"

    df.to_csv(output_path, index=False)
    print(f"Previsões e valores reais salvos em: {output_path}")

def main():
    # Configurações
    config = {
        'file_path': "agentai/datasets/ETTh1.csv",
        'timestamp_column': "date",
        'target_column': "HUFL",
        'prediction_length': 48,  # 48 horas de previsão
        'eval_metric': "MASE",  # Mean Absolute Scaled Error
        'presets': "fast_training",  # Pode usar: fast_training, medium_quality, high_quality, best_quality
        'time_limit': 600,  # 10 minutos
        'num_val_windows': 2,  # Validação mais robusta
        'freq': 'H',  
        'model_path': "AutogluonModels",
    }
    
    print("="*70)
    print("AUTOGLUON TIME SERIES FORECASTING")
    print("="*70)
    print("\nConfigurações:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Executa pipeline
    results = run_complete_forecast(**config)

    # Salva previsões e valores reais em um csv
    save_predictions_and_actuals(config, results)


if __name__ == "__main__":
    main()
