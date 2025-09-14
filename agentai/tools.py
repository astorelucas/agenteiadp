from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
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

from langchain.tools import tool
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.impute import KNNImputer  

from typing import List, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod



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

        imputer = IterativeImputer(
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


# @tool

def analyze_missing_values(df: pd.DataFrame) -> dict:
    """Analyze missing values pattern in time series data"""
    analysis = {
        "total_missing": df.isna().sum().sum(),
        "columns_with_missing": df.columns[df.isna().any()].tolist(),
        "time_gaps": pd.to_datetime(df.index).to_series().diff().value_counts().to_dict()
    }
    return analysis

# # Ferramentas Auxiliares
# @tool
# def salvar_resultados() -> str:
#     """
#     **Salvar Resultados**
#     **Uso ideal**:
#     - Após uma ou mais operações de imputação, normalização, etc terem sido aplicadas e o resultado for satisfatório.
#     - Para persistir o DataFrame processado e evitar a necessidade de reprocessamento.

#     **Custo**: Baixo.

#     **Como age**:
#     - Simplesmente escreve o estado atual do DataFrame em memória para um arquivo CSV no disco, chamado 'resultado_imputado.csv'.
#     - Esta é uma ação final para consolidar as alterações realizadas pelas outras ferramentas.
#     """
#     global df
#     df.to_csv("./datasets/resultado_imputado.csv", index=False)
#     return "DataFrame salvo como 'resultado_imputado.csv'."

# # Inspection Tools
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


# # Cleaning Tools
# @tool
# def clean_data(df_json: str) -> str:
#     """Handle missing values, outliers, and infinity"""
#     try:
#         df = json_to_dataframe(df_json)
        
#         # Replace infinity with NA then interpolate
#         df = df.replace([np.inf, -np.inf], None)
        
#         # Handle missing values
#         if isinstance(df.index, pd.DatetimeIndex):
#             df = df.interpolate(method='time')
#         else:
#             df = df.interpolate()
            
#         # Handle outliers for numeric columns
#         for col in df.select_dtypes(include=['number']).columns:
#             q1 = df[col].quantile(0.25)
#             q3 = df[col].quantile(0.75)
#             iqr = q3 - q1
#             df[col] = df[col].clip(q1-1.5*iqr, q3+1.5*iqr)
            
#         return dataframe_to_json(df)
#     except Exception as e:
#         return dataframe_to_json(pd.DataFrame({"error": [str(e)]}))


# # Download Tool
# @tool
# def save_data(df_json: str) -> str:
#     """Save processed data to CSV with infinity handling"""
#     try:
#         df = json_to_dataframe(df_json)
#         df = df.replace([np.inf, -np.inf], None)
#         filename = f"preprocessed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
#         df.to_csv(filename)
#         return filename
#     except Exception as e:
#         return f"Error saving file: {str(e)}"

def make_plot_tools(df: pd.DataFrame, images_path: str, is_before_dp: bool) -> List:
    """ Create plotting tools with the given DataFrame
    """
    
    # Ajustar o caminho das imagens com base no estado de pré-processamento
    if is_before_dp:
        images_path = images_path + "/before_dp"
    else:
        images_path = images_path + "/after_dp"
    
    # Confirmar que o diretório de imagens existe
    if not os.path.exists(images_path):
        os.makedirs(images_path, exist_ok=True) 

    @tool
    def plot_time_series(cols_str: str = None):
        """
        Plot time series line plot for specified columns with individual subplots.
        If 'cols_str' is None, plots all numeric columns.
        Args:
            cols_str: List of column names to plot (str). If None, plots all numeric columns. Example: "col1,col2,col3"
        Returns:
            dict: Success message or error details
        """
        try:
            # Validar se o DataFrame não está vazio
            if df.empty:
                return {"error": "DataFrame is empty"}

            time_col = df.columns[0]

            if time_col not in df.columns:
                available_cols = list(df.columns)
                return {"error": f"Time column '{time_col}' not found. Available columns: {available_cols}"}

            cols = cols_str.split(",") if cols_str else None

            # Se cols não foi especificado, usar todas as colunas numéricas
            if cols is None:
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if time_col in numeric_cols:
                    numeric_cols.remove(time_col)
                cols = numeric_cols

            if not cols:
                return {"error": "No numeric columns found to plot"}

            # Validar se todas as colunas especificadas existem
            missing_cols = [col for col in cols if col not in df.columns]
            if missing_cols:
                return {"error": f"Columns not found: {missing_cols}"}

            # Verificar se há dados não-nulos para plotar
            valid_data = df[[time_col] + cols].dropna()
            if valid_data.empty:
                return {"error": "No valid data to plot (all values are null)"}

            # Converter a coluna de tempo para datetime
            try:
                time_data = pd.to_datetime(valid_data[time_col])
            except Exception as e:
                return {"error": f"Error converting time column to datetime: {str(e)}"}

            # Filtrar apenas colunas numéricas válidas
            valid_cols = []
            for col in cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    print(f"Warning: Column '{col}' is not numeric, skipping...")
                    continue
                valid_cols.append(col)

            if not valid_cols:
                return {"error": "No valid numeric columns to plot"}

            n_cols = len(valid_cols)
            n_rows = (n_cols + 1) // 2 if n_cols > 1 else 1  # 2 colunas por linha
            n_subplot_cols = min(n_cols, 2)

            # Criar subplots
            fig, axes = plt.subplots(n_rows, n_subplot_cols, figsize=(15, 4 * n_rows))

            if n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes if isinstance(axes, np.ndarray) else [axes]
            else:
                axes = axes.flatten()

            # Plotar cada coluna em seu próprio subplot
            for i, col in enumerate(valid_cols):
                ax = axes[i]
                ax.plot(time_data, df[col], label=col, marker='o', markersize=2, color=f'C{i}')
                ax.set_title(f"Time Series - {col}")
                ax.set_xlabel("Time")
                ax.set_ylabel(col)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                ax.legend()

            if n_cols % 2 == 1 and n_cols > 1:
                axes[-1].set_visible(False)

            plt.suptitle("Time Series Analysis by Column", fontsize=16, y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            plt.savefig(f"{images_path}/time_series_plots.png", bbox_inches="tight", dpi=300)
            plt.close()

            return {"msg": f"Time series subplots created successfully for columns: {valid_cols}"}

        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}


    @tool
    def plot_scatter(two_cols_str: str):
        """
        Create an enhanced scatter plot between two specified columns.

        Args:
            two_cols_str: Comma-separated string of two column names to plot (str). Example: "col1,col2,col3"

        Returns:
            dict: Success message or error details
        """
        try:
            # Validar se o DataFrame não está vazio
            if df.empty:
                return {"error": "DataFrame is empty"}
            
            cols = two_cols_str.split(",") if two_cols_str else None

            # Se cols não foi especificado, retornar erro
            if not cols:
                return {"error": "No columns found to plot"}
            
            # Se cols não contém exatamente 2 colunas, retornar erro
            if len(cols) != 2:
                return {"error": "Please provide exactly two columns for scatter plot"}
            
            x, y = cols
            
            plt.scatter(df[x], df[y])
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(f"{x} vs {y}")
            plt.tight_layout()
            plt.savefig(f"{images_path}/scatter_plot_{x}_vs_{y}.png", bbox_inches="tight", dpi=300)
            plt.close()

            return {"msg": f"Scatter plot created successfully for {x} vs {y}"}

        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    @tool
    def plot_histograms(cols_str: str = None, bins: int = 15):
        """
        Create individual histograms for specified columns.

        Args:
            cols_str: List of column names to plot (str). If None, plots all numeric columns. Example: "col1,col2,col3"
            bins: Number of bins for the histograms (int).

        If 'cols' is None, plot all numeric columns.
        """
        try:

            cols = cols_str.split(",") if cols_str else None
            if cols is None:
                cols = df.select_dtypes(include="number").columns.tolist()

            n_cols = 2  # número de colunas no grid de subplots
            n_rows = (len(cols) + 1) // n_cols

            plt.figure(figsize=(6 * n_cols, 4 * n_rows))

            # Criar subplots
            for idx, col in enumerate(cols, 1):
                plt.subplot(n_rows, n_cols, idx)
                sns.histplot(df[col], bins=bins, kde=True, color="skyblue", edgecolor="black")

                plt.title(f"{col}", fontsize=14)
                plt.xlabel(col, fontsize=12)
                plt.ylabel("Frequency", fontsize=12)
                plt.grid(True, linestyle="--", alpha=0.6)

            plt.suptitle("Histograms of numerical variables", fontsize=16, y=1.02)
            plt.tight_layout()
            plt.savefig(f"{images_path}/histograms.png")
            plt.close()

            return {"msg": "Histograms created successfully."}

        except Exception as e:
            return {"error": str(e)}

    @tool
    def plot_heatmap():
        """
        Create a heatmap of correlations between numeric columns.
        Args:
            None
        Returns:
            dict: Success message or error details
        """
        try:
            plt.figure(figsize=(8, 6))
            corr = df.select_dtypes(include="number").corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(f"{images_path}/correlation_heatmap.png")
            plt.close()

            return {"msg": "Heatmap created successfully."}

        except Exception as e:  
            return {"error": str(e)}
        
    @tool
    def plot_boxplot(cols_str: str = None):
        """
        Create boxplots for specified columns.

        Args:
            cols_str: List of column names to plot (str). If None, plots all numeric columns. Example: "col1,col2,col3"

        If 'cols' is None, plot all numeric columns.
        """
        try:

            cols = cols_str.split(",") if cols_str else None
            if cols is None:
                cols = df.select_dtypes(include="number").columns.tolist()

            n_cols = 2  # número de colunas no grid de subplots
            n_rows = (len(cols) + 1) // n_cols

            plt.figure(figsize=(6 * n_cols, 4 * n_rows))

            # Criar subplots
            for idx, col in enumerate(cols, 1):
                plt.subplot(n_rows, n_cols, idx)
                sns.boxplot(y=df[col], color="lightgreen")

                plt.title(f"{col}", fontsize=14)
                plt.ylabel(col, fontsize=12)
                plt.grid(True, linestyle="--", alpha=0.6)

            plt.suptitle("Boxplots of numerical variables", fontsize=16, y=1.02)
            plt.tight_layout()
            plt.savefig(f"{images_path}/boxplots.png")
            plt.close()

            return {"msg": "Boxplots created successfully."}

        except Exception as e:
            return {"error": str(e)}
        
    @tool
    def plot_scatter_matrix(cols_str: str = None):
        """
        Create a scatter matrix (pair plot) for specified columns.

        Args:
            cols_str: List of column names to plot (str). If None, plots all numeric columns. Example: "col1,col2,col3"

        If 'cols' is None, plot all numeric columns.
        """
        try:

            cols = cols_str.split(",") if cols_str else None
            if cols is None:
                cols = df.select_dtypes(include="number").columns.tolist()

            if len(cols) < 2:
                return {"error": "At least two numeric columns are required for scatter matrix."}

            sns.pairplot(df[cols], diag_kind="kde", plot_kws={"alpha": 0.5})
            plt.suptitle("Scatter Matrix (Pair Plot)", fontsize=16, y=1.02)
            plt.tight_layout()
            plt.savefig(f"{images_path}/scatter_matrix.png")
            plt.close()

            return {"msg": "Scatter matrix created successfully."}

        except Exception as e:
            return {"error": str(e)}
        
    return [plot_time_series, plot_scatter, plot_histograms, plot_heatmap, plot_boxplot, plot_scatter_matrix]


inspection_tools = [inspect_data]
# cleaning_tools = [clean_data]
# feature_tools = [imputacao_k_nearest_neighbors, imputacao_mice, imputacao_gp]
