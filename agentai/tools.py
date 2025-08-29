# from langchain_community.tools import WikipediaQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper
# from langchain.tools import Tool
# from datetime import datetime
import pandas as pd
import json
import numpy as np
import io
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from langchain.tools import tool
# from sklearn.impute import IterativeImputer
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.impute import KNNImputer  

from typing import List, Optional, Tuple, Dict
# import json

# # Tools are things that the LLM/agent can use that we can
# # either write ourself or we can bring in from things like
# # the Langchain Community hub

# @tool
def analyze_missing_values(df: pd.DataFrame) -> dict:
    """Analyze missing values pattern in time series data"""
    analysis = {
        "total_missing": df.isna().sum().sum(),
        "columns_with_missing": df.columns[df.isna().any()].tolist(),
        "time_gaps": pd.to_datetime(df.index).to_series().diff().value_counts().to_dict()
    }
    return analysis

# @tool
# def imputacao_gp() -> str:
#     """
#     **Processos Gaussianos (GP)**

#     **Para que serve**: Modela os dados como uma distribuição de probabilidade, permitindo imputar valores e, crucialmente, estimar a incerteza dessas imputações.

#     **Uso ideal**:
#     - Domínios onde a confiança na imputação é crítica (ex: dados financeiros, médicos, de segurança).
#     - Séries temporais com relações não-lineares complexas.
#     - Datasets de tamanho pequeno a médio.

#     **Pontos Fortes**:
#     - Fornece uma estimativa da incerteza (variância) para cada valor imputado.
#     - Altamente flexível para capturar diferentes tipos de tendências nos dados através de "kernels".

#     **Limitações**:
#     - Custo computacional muito alto, com complexidade cúbica ($O(n^3)$), tornando-o inviável para grandes datasets.
#     - A performance depende muito da escolha do kernel e seus hiperparâmetros.

#     **Como age**:
#     1. Assume que qualquer conjunto de pontos da série temporal segue uma distribuição Gaussiana multivariada.
#     2. "Aprende" os parâmetros dessa distribuição (a função de covariância ou kernel) a partir dos dados observados.
#     3. Usa a distribuição condicionada para prever a média (valor imputado) e a variância (incerteza) dos pontos ausentes.

#     **Custo Computacional**: Muito Alto.
#     """
#     global df
#     imputer = GaussianProcessRegressor(kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)))
#     numeric_cols = df.select_dtypes(include='number').columns

#     if len(numeric_cols) < 2:
#         return "Método GP não aplicado. Requer pelo menos 2 colunas numéricas para usar uma como preditora da outra."

#     try:
#         for col in numeric_cols:
#             if df[col].isnull().any():
#                 feature_cols = numeric_cols.drop(col)
                
#                 observed_idx = df[col].notnull()
#                 missing_idx = df[col].isnull()

#                 if not feature_cols.empty and missing_idx.any():
#                     X_train = df.loc[observed_idx, feature_cols]
#                     y_train = df.loc[observed_idx, col]
#                     X_test = df.loc[missing_idx, feature_cols]

#                     imputer.fit(X_train, y_train)
#                     imputed_values, _ = imputer.predict(X_test, return_std=True)
                    
#                     df.loc[missing_idx, col] = imputed_values

#         return "Processo Gaussiano aplicado. Valores ausentes imputados com base em correlações não-lineares."
#     except Exception as e:
#         return f"Erro ao aplicar Processo Gaussiano: {e}"

# @tool
# def imputacao_mice() -> str:
#     """
#     **MICE (Multivariate Imputation by Chained Equations)**

#     **Para que serve**: Imputa valores ausentes de forma iterativa, usando as outras colunas como preditores, ideal para relações complexas entre variáveis.

#     **Uso ideal**:
#     - Dados ausentes com padrão Aleatório (MAR - Missing at Random), onde a ausência de um valor pode ser explicada por outras variáveis.
#     - Relações lineares ou não-lineares entre as variáveis.
#     - Datasets tabulares multivariados.

#     **Pontos Fortes**:
#     - Muito flexível, pois modela cada coluna condicionalmente.
#     - Pode capturar relações complexas se o estimador for poderoso (como RandomForest).

#     **Limitações**:
#     - O processo é iterativo e pode ser lento para datasets grandes ou com muitos ciclos.
#     - Assume que uma relação condicional existe entre as variáveis, o que pode não ser verdade.

#     **Como age**:
#     1. Preenche temporariamente todos os `NaN`s com uma estimativa simples (ex: média).
#     2. Para cada variável (coluna), define-a como a variável alvo e as outras como preditoras.
#     3. Treina um modelo (neste caso, `RandomForestRegressor`) para prever os valores originalmente ausentes naquela coluna.
#     4. Repete o passo 3 para todas as colunas em ciclos, refinando as imputações até que os valores se estabilizem.

#     **Custo Computacional**: Médio.
#     """
#     global df
#     df_numeric = df.select_dtypes(include='number').copy()

#     for col in df_numeric.columns:
#         df_numeric[f'{col}_lag1'] = df_numeric[col].shift(1)

#     imputer = IterativeImputer(
#         estimator=RandomForestRegressor(n_estimators=10),
#         random_state=0
#     )
#     imputed_matrix = imputer.fit_transform(df_numeric)
#     df_imputed_temp = pd.DataFrame(imputed_matrix, columns=df_numeric.columns, index=df_numeric.index)

#     original_numeric_cols = df.select_dtypes(include='number').columns
#     df[original_numeric_cols] = df_imputed_temp[original_numeric_cols]

#     return "MICE aplicado. Relações entre colunas e lags temporais foram consideradas."

# @tool
# def imputacao_k_nearest_neighbors(k=5): 
#     """
#     **K-Nearest Neighbors (KNN)**
#     **Uso ideal**:
#     - Dados com padrões locais, onde valores próximos no tempo ou no espaço de características são semelhantes.
#     - Missing Data Completamente Aleatório (MCAR) ou Aleatório (MAR).
#     - Datasets de pequeno a médio porte, onde a busca por vizinhos não é computacionalmente proibitiva.

#     **Exemplo**: Imputar a temperatura de uma estação meteorológica baseando-se nos registros de horas ou dias próximos.

#     **Custo**: Baixo a Médio.

#     **Como age**:
#     1. Para cada ponto de dados ausente, o algoritmo identifica os 'k' vizinhos mais próximos no espaço de características (outras variáveis).
#     2. O valor ausente é imputado calculando a média (ou outra métrica) dos valores desses 'k' vizinhos.
#     3. A premissa é que um ponto de dados provavelmente será semelhante aos seus vizinhos mais próximos.
#     """
#     global df
#     df_nulls = df.copy().drop('timestamp', axis=1).select_dtypes(include='number')
#     imputer = KNNImputer(n_neighbors=k)
#     df_filled = pd.DataFrame(imputer.fit_transform(df_nulls),
#                              columns=df_nulls.columns,
#                              index=df_nulls.index)    
#     df[df_filled.columns] = df_filled 

#     return "KNN aplicado ao dataframe original."

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



inspection_tools = [inspect_data]
# cleaning_tools = [clean_data]
# feature_tools = [imputacao_k_nearest_neighbors, imputacao_mice, imputacao_gp]