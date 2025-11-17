from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool

from langchain.tools import tool
import pandas as pd
import json
import numpy as np
import io
import os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.impute import KNNImputer  

from typing import List, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
from agentai.rag import RAG

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# abstract class 
class ImputationStrategy(ABC):
    @abstractmethod
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class GPImputationStrategy(ImputationStrategy):
    def __init__(self, kernel=None):
        self.kernel = kernel or C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        df_final = df.copy()
        numeric_cols = df_final.select_dtypes(include='number').columns

        if len(numeric_cols) < 2:
            print("GP method not applied. Requires at least 2 numeric columns.")
            return df_final

        imputer_gp = GaussianProcessRegressor(kernel=self.kernel)
        
        for col_to_impute in list(numeric_cols):
            if df_final[col_to_impute].isnull().any():
                
                feature_cols = numeric_cols.drop(col_to_impute)
                observed_idx = df_final[col_to_impute].notnull()
                missing_idx = df_final[col_to_impute].isnull()

                if feature_cols.empty or not missing_idx.any():
                    continue

                X_observed = df_final.loc[observed_idx, feature_cols]
                y_observed = df_final.loc[observed_idx, col_to_impute]
                X_missing = df_final.loc[missing_idx, feature_cols]

                if X_observed.isnull().values.any() or X_missing.isnull().values.any():
                    pre_imputer_knn = KNNImputer(n_neighbors=5)
                    
                    X_observed_imputed = pd.DataFrame(pre_imputer_knn.fit_transform(X_observed), columns=feature_cols, index=X_observed.index)
                    X_missing_imputed = pd.DataFrame(pre_imputer_knn.transform(X_missing), columns=feature_cols, index=X_missing.index)
                else:
                    X_observed_imputed = X_observed
                    X_missing_imputed = X_missing

                imputer_gp.fit(X_observed_imputed, y_observed)
                
                imputed_values, _ = imputer_gp.predict(X_missing_imputed, return_std=True)
                df_final.loc[missing_idx, col_to_impute] = imputed_values
        
        print("Robust Gaussian Process strategy executed successfully.")
        return df_final

class MICEImputationStrategy(ImputationStrategy):
    """
    Performs MICE (Multivariate Imputation by Chained Equations) on a DataFrame.
    This function automatically isolates numeric columns, applies imputation using RandomForestRegressor, 
    and then reintegrates the original non-numeric columns.
    The function MUST be called with the complete DataFrame as an argument (e.g., imputacao_mice(df)).
    It returns the complete DataFrame with the imputed numeric values.
    """
    def __init__(self, n_estimators: int = 10, random_state: int = 0):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        df_final = df.copy()
        numeric_cols = df_final.select_dtypes(include='number').columns
        if len(numeric_cols) == 0:
            return df_final
        
        df_numeric = df_final[numeric_cols].copy()
        for col in df_numeric.columns:
            df_numeric[f'{col}_lag1'] = df_numeric[col].shift(1)

        imputer = enable_iterative_imputer(
            estimator=RandomForestRegressor(n_estimators=self.n_estimators),
            random_state=self.random_state
        )
        imputed_matrix = imputer.fit_transform(df_numeric)
        df_imputed_temp = pd.DataFrame(imputed_matrix, columns=df_numeric.columns, index=df_numeric.index)
        df_final[numeric_cols] = df_imputed_temp[numeric_cols]
        print("MICE strategy executed.")
        return df_final

class KNNImputationStrategy(ImputationStrategy):
    """
    Performs K-Nearest Neighbors (KNN) imputation on a DataFrame.
    This method is ideal for datasets with local patterns where similar data points have similar values (e.g., sensor or spatial data).
    It is best used on small to medium-sized datasets and when data is Missing Completely at Random (MCAR) or at Random (MAR).
    For each missing value, it finds the 'k' most similar records and imputes the value based on their average (or median/mode).
    It returns the complete DataFrame with the imputed values.
    """
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        df_final = df.copy()
        numeric_cols = df_final.select_dtypes(include='number').columns
        if len(numeric_cols) == 0:
            return df_final

        df_numeric = df_final[numeric_cols]
        imputer = KNNImputer(n_neighbors=self.n_neighbors)
        df_filled_matrix = imputer.fit_transform(df_numeric)
        df_filled = pd.DataFrame(df_filled_matrix, columns=numeric_cols, index=df_numeric.index)
        df_final[numeric_cols] = df_filled
        print("KNN strategy executed.")
        return df_final

# factory strategy :)
class ImputationStrategyFactory:
    _strategies = {
        "gp": GPImputationStrategy,
        "mice": MICEImputationStrategy,
        "knn": KNNImputationStrategy,
    }

    def create_strategy(self, name: str, **kwargs: Any) -> ImputationStrategy:
        strategy_class = self._strategies.get(name)
        if not strategy_class:
            raise ValueError(f"'{name}' strategy not recognized")
        try:
            return strategy_class(**kwargs)
        except TypeError as e:
            raise TypeError(f"Invalid parameters for '{name}': {e}")

def analyze_missing_values(df: pd.DataFrame) -> dict:
    """Analyze missing values pattern in time series data"""
    analysis = {
        "total_missing": df.isna().sum().sum(),
        "columns_with_missing": df.columns[df.isna().any()].tolist(),
        "time_gaps": pd.to_datetime(df.index).to_series().diff().value_counts().to_dict()
    }
    return analysis

# Inspection Tools
@tool
def inspect_data(df: str) -> Dict:
    """Perform a comprehensive inspection of a time series DataFrame."""
    try:
        if isinstance(df, str):
            # Try to convert from CSV string
            df = pd.read_csv(io.StringIO(df))  # or use 
        # Replace infinity
                # Replace infinity values with None
        df_clean = df.replace([np.inf, -np.inf], None)

        # Descriptive stats for all numeric columns
        stats = df_clean.describe(include='all')
        stats_dict = json.loads(stats.to_json())

        missing_values = df_clean.isna().sum().to_dict()

        has_infinity = bool(df_clean.isin([np.inf, -np.inf]).any().any())

        return {
            "missing_values": analyze_missing_values(df),
            "statistics": stats_dict,
            "has_infinity": has_infinity
        }
    except Exception as e:
        return {"error": str(e)}

# def make_plot_tools(df: pd.DataFrame, images_path: str, is_before_dp: bool) -> List:
#     """ Create plotting tools with the given DataFrame
#     """
    
#     images_path = images_path
    
#     if not os.path.exists(images_path):
#         os.makedirs(images_path, exist_ok=True) 

#     @tool
#     def plot_time_series(cols_str: str = None):
#         """
#         Plot time series line plot for specified columns with individual subplots.
#         If 'cols_str' is None, plots all numeric columns.
#         Args:
#             cols_str: List of column names to plot (str). If None, plots all numeric columns. Example: "col1,col2,col3"
#         Returns:
#             dict: Success message or error details
#         """
#         try:
#             # Validar se o DataFrame não está vazio
#             if df.empty:
#                 return {"error": "DataFrame is empty"}

#             time_col = df.columns[0]

#             if time_col not in df.columns:
#                 available_cols = list(df.columns)
#                 return {"error": f"Time column '{time_col}' not found. Available columns: {available_cols}"}

#             cols = cols_str.split(",") if cols_str else None

#             # Se cols não foi especificado, usar todas as colunas numéricas
#             if cols is None:
#                 numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
#                 if time_col in numeric_cols:
#                     numeric_cols.remove(time_col)
#                 cols = numeric_cols

#             if not cols:
#                 return {"error": "No numeric columns found to plot"}

#             # Validar se todas as colunas especificadas existem
#             missing_cols = [col for col in cols if col not in df.columns]
#             if missing_cols:
#                 return {"error": f"Columns not found: {missing_cols}"}

#             # Verificar se há dados não-nulos para plotar
#             valid_data = df[[time_col] + cols].dropna()
#             if valid_data.empty:
#                 return {"error": "No valid data to plot (all values are null)"}

#             # Converter a coluna de tempo para datetime
#             try:
#                 time_data = pd.to_datetime(valid_data[time_col])
#             except Exception as e:
#                 return {"error": f"Error converting time column to datetime: {str(e)}"}

#             # Filtrar apenas colunas numéricas válidas
#             valid_cols = []
#             for col in cols:
#                 if not pd.api.types.is_numeric_dtype(df[col]):
#                     print(f"Warning: Column '{col}' is not numeric, skipping...")
#                     continue
#                 valid_cols.append(col)

#             if not valid_cols:
#                 return {"error": "No valid numeric columns to plot"}

#             n_cols = len(valid_cols)
#             n_rows = (n_cols + 1) // 2 if n_cols > 1 else 1  # 2 colunas por linha
#             n_subplot_cols = min(n_cols, 2)

#             # Criar subplots
#             fig, axes = plt.subplots(n_rows, n_subplot_cols, figsize=(15, 4 * n_rows))

#             if n_cols == 1:
#                 axes = [axes]
#             elif n_rows == 1:
#                 axes = axes if isinstance(axes, np.ndarray) else [axes]
#             else:
#                 axes = axes.flatten()

#             # Plotar cada coluna em seu próprio subplot
#             for i, col in enumerate(valid_cols):
#                 ax = axes[i]
#                 ax.plot(time_data, valid_data[col], label=col, marker='o', markersize=2, color=f'C{i}')
#                 ax.set_title(f"Time Series - {col}")
#                 ax.set_xlabel("Time")
#                 ax.set_ylabel(col)
#                 ax.tick_params(axis='x', rotation=45)
#                 ax.grid(True, alpha=0.3)
#                 ax.legend()

#             if n_cols % 2 == 1 and n_cols > 1:
#                 axes[-1].set_visible(False)

#             plt.suptitle("Time Series Analysis by Column", fontsize=16, y=0.98)
#             plt.tight_layout()
#             os.makedirs(images_path, exist_ok=True)
#             plt.savefig(os.path.join(images_path, "time_series_plots.png"), bbox_inches="tight", dpi=300)

#             plt.close()

#             return {"msg": f"Time series subplots created successfully for columns: {valid_cols}"}

#         except Exception as e:
#             return {"error": f"Unexpected error: {str(e)}"}


#     @tool
#     def plot_scatter(two_cols_str: str):
#         """
#         Create an enhanced scatter plot between two specified columns.

#         Args:
#             two_cols_str: Comma-separated string of two column names to plot (str). Example: "col1,col2,col3"

#         Returns:
#             dict: Success message or error details
#         """
#         try:
#             # Validar se o DataFrame não está vazio
#             if df.empty:
#                 return {"error": "DataFrame is empty"}
            
#             cols = two_cols_str.split(",") if two_cols_str else None

#             # Se cols não foi especificado, retornar erro
#             if not cols:
#                 return {"error": "No columns found to plot"}
            
#             # Se cols não contém exatamente 2 colunas, retornar erro
#             if len(cols) != 2:
#                 return {"error": "Please provide exactly two columns for scatter plot"}
            
#             x, y = cols
            
#             plt.scatter(df[x], df[y])
#             plt.xlabel(x)
#             plt.ylabel(y)
#             plt.title(f"{x} vs {y}")
#             plt.tight_layout()
#             os.makedirs(images_path, exist_ok=True)
#             filename = f"scatter_{x}_vs_{y}.png"
#             plt.savefig(os.path.join(images_path, filename), bbox_inches="tight", dpi=300)
#             plt.close()

#             return {"msg": f"Scatter plot created successfully for {x} vs {y}"}

#         except Exception as e:
#             return {"error": f"Unexpected error: {str(e)}"}
    
#     @tool
#     def plot_histograms(cols_str: str = None, bins: int = 15):
#         """
#         Create individual histograms for specified columns.

#         Args:
#             cols_str: List of column names to plot (str). If None, plots all numeric columns. Example: "col1,col2,col3"
#             bins: Number of bins for the histograms (int).

#         If 'cols' is None, plot all numeric columns.
#         """
#         try:

#             cols = cols_str.split(",") if cols_str else None
#             if cols is None:
#                 cols = df.select_dtypes(include="number").columns.tolist()

#             n_cols = 2  # número de colunas no grid de subplots
#             n_rows = (len(cols) + 1) // n_cols

#             plt.figure(figsize=(6 * n_cols, 4 * n_rows))

#             # Criar subplots
#             for idx, col in enumerate(cols, 1):
#                 plt.subplot(n_rows, n_cols, idx)
#                 sns.histplot(df[col], bins=bins, kde=True, color="skyblue", edgecolor="black")

#                 plt.title(f"{col}", fontsize=14)
#                 plt.xlabel(col, fontsize=12)
#                 plt.ylabel("Frequency", fontsize=12)
#                 plt.grid(True, linestyle="--", alpha=0.6)

#             plt.suptitle("Histograms of numerical variables", fontsize=16, y=1.02)
#             plt.tight_layout()
#             os.makedirs(images_path, exist_ok=True)
#             filename = f"histogram_{col}.png"
#             plt.savefig(os.path.join(images_path, filename), bbox_inches="tight", dpi=300)

#             plt.close()

#             return {"msg": "Histograms created successfully."}

#         except Exception as e:
#             return {"error": str(e)}

#     @tool
#     def plot_heatmap():
#         """
#         Create a heatmap of correlations between numeric columns.
#         Args:
#             None
#         Returns:
#             dict: Success message or error details
#         """
#         try:
#             plt.figure(figsize=(8, 6))
#             corr = df.select_dtypes(include="number").corr()
#             sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
#             plt.title("Correlation Heatmap")
#             plt.tight_layout()
#             os.makedirs(images_path, exist_ok=True)
#             plt.savefig(os.path.join(images_path, "heatmap.png"), bbox_inches="tight", dpi=300)

#             plt.close()

#             return {"msg": "Heatmap created successfully."}

#         except Exception as e:  
#             return {"error": str(e)}
        
#     @tool
#     def plot_boxplot(cols_str: str = None):
#         """
#         Create boxplots for specified columns.

#         Args:
#             cols_str: List of column names to plot (str). If None, plots all numeric columns. Example: "col1,col2,col3"

#         If 'cols' is None, plot all numeric columns.
#         """
#         try:

#             cols = cols_str.split(",") if cols_str else None
#             if cols is None:
#                 cols = df.select_dtypes(include="number").columns.tolist()

#             n_cols = 2  # número de colunas no grid de subplots
#             n_rows = (len(cols) + 1) // n_cols

#             plt.figure(figsize=(6 * n_cols, 4 * n_rows))

#             # Criar subplots
#             for idx, col in enumerate(cols, 1):
#                 plt.subplot(n_rows, n_cols, idx)
#                 sns.boxplot(y=df[col], color="lightgreen")

#                 plt.title(f"{col}", fontsize=14)
#                 plt.ylabel(col, fontsize=12)
#                 plt.grid(True, linestyle="--", alpha=0.6)

#             plt.suptitle("Boxplots of numerical variables", fontsize=16, y=1.02)
#             plt.tight_layout()
#             plt.savefig(f"{images_path}/boxplots.png")
#             plt.close()

#             return {"msg": "Boxplots created successfully."}

#         except Exception as e:
#             return {"error": str(e)}
        
#     @tool
#     def plot_scatter_matrix(cols_str: str = None):
#         """
#         Create a scatter matrix (pair plot) for specified columns.

#         Args:
#             cols_str: List of column names to plot (str). If None, plots all numeric columns. Example: "col1,col2,col3"

#         If 'cols' is None, plot all numeric columns.
#         """
#         try:

#             cols = cols_str.split(",") if cols_str else None
#             if cols is None:
#                 cols = df.select_dtypes(include="number").columns.tolist()

#             if len(cols) < 2:
#                 return {"error": "At least two numeric columns are required for scatter matrix."}

#             sns.pairplot(df[cols], diag_kind="kde", plot_kws={"alpha": 0.5})
#             plt.suptitle("Scatter Matrix (Pair Plot)", fontsize=16, y=1.02)
#             plt.tight_layout()
#             plt.savefig(f"{images_path}/scatter_matrix.png")
#             plt.close()

#             return {"msg": "Scatter matrix created successfully."}

#         except Exception as e:
#             return {"error": str(e)}
        
#     return [plot_time_series, plot_scatter, plot_histograms, plot_heatmap, plot_boxplot, plot_scatter_matrix]

def make_automl_tools(df: pd.DataFrame, target: str, test_size: float = 0.2, prediction_length: int = 1, eval_metric: str = "MASE") -> List:
    """ Create AutoML tools with the given DataFrame
    """

    # ========== Helper Functions ==========
    # ideia principal é modularizar a tool para reutilização e clareza
    # Ainda vou fazer (TODO): tornar o forecast também modularizado e criar classes para as métricas (MAE, RMSE, MAPE, etc)
    
    
    def _save_predictions_csv(forecast_timestamps, forecast_actuals, forecast_preds, preds, output_path, logs):
        """Save predictions and actuals to CSV file."""
        try:
            timestamps_str = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in forecast_timestamps]
            comparison_df = pd.DataFrame({
                'timestamp': timestamps_str,
                'actual': forecast_actuals,
                'predicted': forecast_preds,
                'lower_bound': preds['0.1'].values,
                'upper_bound': preds['0.9'].values
            })

            csv_path = os.path.join(output_path, "predictions_and_actuals.csv")
            comparison_df.to_csv(csv_path, index=False)
            logs.append(f"Saved predictions to {csv_path}")
            return True
        except Exception as e:
            logs.append(f"CSV save failed: {e}")
            return False

    def _plot_main_forecast(train_timestamps, train_values, forecast_timestamps, forecast_actuals, 
                           forecast_preds, preds, target, output_path, logs):
        """Create main time series forecast plot with train and forecast periods."""
        try:
            fig, ax = plt.subplots(figsize=(20, 6))
            
            # Converter timestamps para listas
            train_timestamps_list = train_timestamps.to_list()
            forecast_timestamps_list = forecast_timestamps.to_list()
            
            # Plot treino
            ax.plot(train_timestamps_list, train_values, 
                    color='blue', linewidth=1.5, alpha=0.7, label="Training data")
            
            # Criar arrays estendidos para conectar treino com forecast
            extended_forecast_timestamps = [train_timestamps_list[-1]] + forecast_timestamps_list
            extended_actuals = [train_values[-1]] + list(forecast_actuals)
            extended_preds = [train_values[-1]] + list(forecast_preds)
            
            # Plot valores reais do forecast
            ax.plot(extended_forecast_timestamps, extended_actuals, 
                    color='green', linewidth=2, marker='o', markersize=4,
                    label="Actual values (forecast period)")
            
            # Plot previsões
            ax.plot(extended_forecast_timestamps, extended_preds, 
                    color='red', linewidth=2, marker='x', markersize=5,
                    label="Forecast", linestyle='--')

            # Linha vertical de separação
            split_time = train_timestamps[-1]
            ax.axvline(x=split_time, color='black', linestyle='--', 
                      linewidth=1.5, alpha=0.6, label='Train/Forecast split')

            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel(target, fontsize=12)
            ax.set_title("AutoGluon Time Series Forecast", fontsize=14)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45, ha='right')
            ax.set_xlim([train_timestamps[0], forecast_timestamps[-1]])
            
            plt.tight_layout()
            fig_path = os.path.join(output_path, "prediction_plot.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()

            logs.append(f"Saved forecast plot to {fig_path}")
            return True
        except Exception as e:
            logs.append(f"Main forecast plot failed: {e}")
            return False

    def _plot_forecast_vs_actual(
        forecast_timestamps, forecast_actuals, forecast_preds, 
        preds, target, output_path, logs
    ):
        """Create isolated forecast period comparison plot (only one figure)."""
        try:
            # Clear any previous figures
            plt.close('all')

            fig, ax = plt.subplots(figsize=(12, 6))

            forecast_timestamps_list = forecast_timestamps.to_list()

            ax.plot(
                forecast_timestamps_list,
                forecast_actuals,
                color='green', linewidth=2, marker='o', markersize=5,
                label="Actual values"
            )

            ax.plot(
                forecast_timestamps_list,
                forecast_preds,
                color='red', linewidth=2, marker='x', markersize=5,
                label="Forecast", linestyle='--'
            )

            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel(target, fontsize=12)
            ax.set_title("Forecast Period: Predicted vs Actual", fontsize=14)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45, ha='right')

            plt.tight_layout()

            forecast_plot_path = os.path.join(output_path, "forecast_vs_actual.png")
            fig.savefig(forecast_plot_path, dpi=300, bbox_inches='tight')

            plt.close(fig)

            logs.append(f"Saved forecast vs actual plot to {forecast_plot_path}")
            return True
        
        except Exception as e:
            logs.append(f"Forecast vs actual plot failed: {e}")
            return False

    def _plot_error_histogram(forecast_actuals, forecast_preds, output_path, logs):
        """Create error distribution histogram with statistics."""
        try:
            errors = forecast_preds - forecast_actuals
            
            fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
            
            n, bins, patches = ax_hist.hist(errors, bins=20, color='skyblue', 
                                            edgecolor='black', alpha=0.7, density=True)
            
            # Curva normal
            mu, std = errors.mean(), errors.std()
            from scipy.stats import norm
            xmin, xmax = ax_hist.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax_hist.plot(x, p, 'r-', linewidth=2, label=f'Normal dist. (μ={mu:.2f}, σ={std:.2f})')
            
            # Métricas
            mae = np.abs(errors).mean()
            rmse = np.sqrt((errors**2).mean())
            
            ax_hist.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Perfect forecast')
            ax_hist.axvline(x=mu, color='red', linestyle='--', linewidth=2, label=f'Mean error: {mu:.2f}')
            
            ax_hist.set_xlabel("Prediction Error", fontsize=12)
            ax_hist.set_ylabel("Density", fontsize=12)
            ax_hist.set_title(f"Error Distribution (MAE={mae:.2f}, RMSE={rmse:.2f})", fontsize=14)
            ax_hist.legend(loc='best', fontsize=10)
            ax_hist.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            hist_path = os.path.join(output_path, "error_histogram.png")
            plt.savefig(hist_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logs.append(f"Saved error histogram to {hist_path}")
            return True
        except Exception as e:
            logs.append(f"Error histogram failed: {e}")
            return False

    def _plot_error_over_time(forecast_timestamps, forecast_actuals, forecast_preds, output_path, logs):
        """Create error over time plot with absolute and percentage errors."""
        try:
            errors = forecast_preds - forecast_actuals
            forecast_timestamps_list = forecast_timestamps.to_list()
            
            fig_error, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Erros absolutos
            ax1.plot(forecast_timestamps_list, errors, color='blue', linewidth=1.5, 
                    marker='o', markersize=4, label='Prediction error')
            ax1.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Perfect forecast')
            
            if len(errors) >= 3:
                window = min(5, len(errors) // 3)
                ma = pd.Series(errors).rolling(window=window, center=True).mean()
                ax1.plot(forecast_timestamps_list, ma, color='red', linewidth=2, 
                        linestyle='--', label=f'Moving avg (window={window})')
            
            ax1.set_xlabel("Time", fontsize=12)
            ax1.set_ylabel("Error", fontsize=12)
            ax1.set_title("Prediction Error Over Time", fontsize=14)
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Erros percentuais
            pct_errors = 100 * errors / forecast_actuals
            ax2.plot(forecast_timestamps_list, pct_errors, color='purple', linewidth=1.5,
                    marker='s', markersize=4, label='Percentage error')
            ax2.axhline(y=0, color='green', linestyle='--', linewidth=2)
            
            if len(pct_errors) >= 3:
                window = min(5, len(pct_errors) // 3)
                ma_pct = pd.Series(pct_errors).rolling(window=window, center=True).mean()
                ax2.plot(forecast_timestamps_list, ma_pct, color='red', linewidth=2,
                        linestyle='--', label=f'Moving avg (window={window})')
            
            ax2.set_xlabel("Time", fontsize=12)
            ax2.set_ylabel("Error (%)", fontsize=12)
            ax2.set_title("Percentage Error Over Time", fontsize=14)
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            error_time_path = os.path.join(output_path, "error_over_time.png")
            plt.savefig(error_time_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logs.append(f"Saved error over time plot to {error_time_path}")
            return True
        except Exception as e:
            logs.append(f"Error over time plot failed: {e}")
            return False

    # ========== Main Tool ==========

    def autogluon_forecast(input_text: str) -> dict:
        """
        Perform time series forecasting using AutoGluon TimeSeriesPredictor on the in-memory DataFrame.
        Uses the first time/date-like column as timestamp and a single series item_id.
        Returns real values, forecast values, best model info, evaluation metrics and basic logs.
        """
        logs = []
        new_df = df.copy()

        # ---------- Validation ----------
        if not isinstance(new_df, pd.DataFrame):
            return {"error": "Input 'df' must be a pandas DataFrame.", "logs": logs}
        if new_df.empty:
            return {"error": "Dataset is empty.", "logs": logs}
        if not isinstance(target, str) or target not in new_df.columns:
            return {"error": f"Target column '{target}' not found in dataset.", "logs": logs}
        if not isinstance(test_size, float) or not (0 < test_size < 1):
            return {"error": "Invalid 'test_size'. Must be a float between 0 and 1.", "logs": logs}

        # ---------- Lazy import ----------
        try:
            from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor  # type: ignore
        except Exception as e:
            return {"error": f"AutoGluon not available: {e}. Try: pip install autogluon.timeseries", "logs": logs}

        # ---------- Identify timestamp column ----------
        time_cols = [c for c in new_df.columns if ("time" in c.lower()) or ("date" in c.lower()) or ("stamp" in c.lower())]
        if not time_cols:
            return {"error": "No time/date column inferred. Ensure there is a timestamp column.", "logs": logs}
        time_col = time_cols[0]
        logs.append(f"Using '{time_col}' as timestamp column.")

        # ---------- Prepare TimeSeriesDataFrame ----------
        try:
            # --- Prepare DataFrame ---
            new_df[time_col] = pd.to_datetime(new_df[time_col])
            new_df = new_df.sort_values(time_col)

            # AutoGluon requires item_id even for 1 series
            new_df["item_id"] = "series_0"

            # Keep ALL columns (target + covariates)
            all_cols = ["item_id", time_col] + [c for c in new_df.columns if c not in ["item_id", time_col]]

            # NEW: DO NOT convert to TimeSeriesDataFrame before splitting
            # Because AG splits must receive the *same set of feature columns*

            # --- Train-test split ---
            train_fraction = 1 - test_size
            n = len(new_df)

            min_length_allowed = prediction_length + 1
            if n < min_length_allowed:
                raise ValueError(f"Series too short: length {n}, needs at least {min_length_allowed}")

            split_idx = int(n * train_fraction)
            if split_idx <= prediction_length:
                split_idx = prediction_length + 1

            train_df = new_df.iloc[:split_idx].copy()
            test_df  = new_df.iloc[split_idx:].copy()
            
            # --- Convert to AutoGluon TimeSeriesDataFrame ---
            train_data = TimeSeriesDataFrame.from_data_frame(
                train_df[all_cols],
                id_column="item_id",
                timestamp_column=time_col
            )  

            test_data = TimeSeriesDataFrame.from_data_frame(
                test_df[all_cols],
                id_column="item_id",
                timestamp_column=time_col
            )

        except Exception as e:
            return {"error": f"Failed to build TimeSeriesDataFrame: {e}", "logs": logs}


        # ---------- Split ----------
        # try:
        #     train_data, test_data = tsdf.train_test_split(train_fraction=1 - test_size)
        #     logs.append(f"Split data into train ({len(train_data)}) and test ({len(test_data)}). Prediction length={fh}")
        # except Exception as e:
        #     return {"error": f"Failed to split data: {e}", "logs": logs}
        
        # ---------- Frequency inference ----------
        try:
            freq = pd.infer_freq(train_data.index.get_level_values('timestamp'))
            if freq is None:
                freq = 'H'  # default to hourly if inference fails
            logs.append(f"Inferred data frequency: {freq}")
        except Exception as e:
            return {"error": f"Frequency inference failed: {e}", "logs": logs}

        # ---------- Train ----------
        try:
            predictor = TimeSeriesPredictor(
                prediction_length=prediction_length,
                target=target,
                eval_metric=eval_metric,
                verbosity=2,
                freq=freq
            )
            predictor.fit(
                train_data=train_data,
                tuning_data=test_data,
                presets="high_quality",
                time_limit=None,
                hyperparameter_tune_kwargs=None
            )
            logs.append("AutoGluon training completed using 'high_quality' presets.")
        except Exception as e:
            return {"error": f"AutoGluon training failed: {e}", "logs": logs}

        # ---------- Predict ----------
        try:
            preds = predictor.predict(test_data)
            mean_series = preds["mean"]
            real_series = test_data[target]
            real_tail = real_series.groupby(level="item_id").tail(prediction_length).values
            forecast_vals = mean_series.groupby(level="item_id").tail(prediction_length).values
        except Exception as e:
            return {"error": f"Prediction failed: {e}", "logs": logs}

        # ---------- Best Model ----------
        try:
            best_model = predictor.model_best
        except Exception:
            best_model = None

        # ---------- Extract data for visualization ----------
        train_timestamps = train_data.index.get_level_values('timestamp')
        train_values = train_data[target].values
        forecast_timestamps = test_data.index.get_level_values('timestamp')[-prediction_length:]
        forecast_actuals = test_data[target].iloc[-prediction_length:].values
        forecast_preds = preds['mean'].values

        # ---------- Output path definition ----------
        output_path = "agentai/results/forecast/test/" 
        os.makedirs(output_path, exist_ok=True)
        logs.append(f"Output path set to: {output_path}")

        # ---------- Save CSV ----------
        _save_predictions_csv(forecast_timestamps, forecast_actuals, forecast_preds, preds, output_path, logs)

        # ---------- Generate Plots ----------
        # _plot_main_forecast(train_timestamps, train_values, forecast_timestamps, 
        #                     forecast_actuals, forecast_preds, preds, target, output_path, logs)
        
        _plot_forecast_vs_actual(forecast_timestamps, forecast_actuals, forecast_preds, 
                                preds, target, output_path, logs)
        
        # _plot_error_histogram(forecast_actuals, forecast_preds, output_path, logs)
        
        # _plot_error_over_time(forecast_timestamps, forecast_actuals, forecast_preds, output_path, logs)

        # ---------- Compute Metrics ----------
        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            import numpy as np

            mae = mean_absolute_error(forecast_actuals, forecast_preds)
            rmse = np.sqrt(mean_squared_error(forecast_actuals, forecast_preds))
            mape = np.mean(np.abs((forecast_actuals - forecast_preds) / forecast_actuals)) * 100

            # Compute MASE via AutoGluon's built-in evaluation
            eval_results = predictor.evaluate(test_data)
            mase = eval_results.get("MASE", None) if isinstance(eval_results, dict) else None

            metrics = {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "MAPE": float(mape),
                "MASE": float(mase) if mase is not None else None
            }

            logs.append(f"Evaluation metrics computed: {metrics}")
        except Exception as e:
            metrics = {"error": f"Metrics computation failed: {e}"}
            logs.append(f"Metrics computation failed: {e}")

        # ---------- Return ----------
        return {
            "best_model": str(best_model) if best_model is not None else None,
            "metrics": metrics,
            "logs": logs,
        }


    return [
        Tool.from_function(
            autogluon_forecast,
            name="autogluon_forecast",
            description="Run AutoGluon time-series forecasting on the provided DataFrame and return a JSON object with keys like best_model, logs, metrics and file paths."
        )
    ]


@tool
def retrieve_context(query: str) -> dict:
    """
    Retrieves relevant context from the vector database using a RAG pipeline.
    This function acts as a knowledge-base tool for agents, allowing them to fetch information to ground their responses or inform their actions.
    *ALWAYS* use it when having errors or doubts or even if you are unsure about any future action!
    
    Args:
        query: The natural language question or topic to search for.
    Returns:
        A dictionary containing the retrieved documents and associated metadata.
    """
    rag = RAG()
    return rag.retrieve(query)

inspection_tools = [inspect_data]