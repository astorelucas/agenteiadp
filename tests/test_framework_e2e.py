"""
Framework de Teste End-to-End para Agente Multiagente
Versão Corrigida e Otimizada - Avalia o desempenho de cada nó
"""

import pandas as pd
import numpy as np
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib
matplotlib.use('Agg')  # Backend não interativo
import matplotlib.pyplot as plt
import seaborn as sns


class AgentNodeAnalyzer:
    """Analisa as ações e resultados de cada nó do agente"""
    
    def __init__(self):
        self.results = {
            'inspector': {},
            'imputator': {},
            'feature_engineer': {},
            'plotter': {},
            'supervisor': {},
            'automl': {}
        }
    
    def analyze_inspector_node(self, logs: List[str], df_original: pd.DataFrame, 
                               df_after_inspection: pd.DataFrame) -> Dict:
        """
        Analisa o Nó Inspetor
        Retorna:
        - % faltantes identificados
        - Características extraídas (shape, tipos, estatísticas)
        """
        analysis = {
            'missing_percentage_identified': None,
            'characteristics_extracted': {},
            'errors': []
        }
        
        # Calcular % de valores faltantes real
        total_cells = df_original.size
        missing_cells = df_original.isna().sum().sum()
        real_missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # Procurar nos logs se identificou valores faltantes
        inspector_logs = [log for log in logs if '[Pandas Node]' in log or 'inspect' in log.lower()]
        
        identified_missing = False
        for log in inspector_logs:
            if 'missing' in log.lower() or 'null' in log.lower() or 'nan' in log.lower():
                identified_missing = True
                # Tentar extrair percentual mencionado
                numbers = re.findall(r'(\d+\.?\d*)%', log)
                if numbers:
                    analysis['missing_percentage_identified'] = float(numbers[0])
                break
        
        if not identified_missing and missing_cells > 0:
            analysis['errors'].append("Valores faltantes existem mas não foram identificados")
        
        # Se não encontrou nos logs, assumir que identificou corretamente
        if identified_missing and analysis['missing_percentage_identified'] is None:
            analysis['missing_percentage_identified'] = real_missing_pct
        
        # Verificar características extraídas
        characteristics = {
            'shape_identified': False,
            'dtypes_identified': False,
            'statistics_computed': False,
            'columns_listed': False
        }
        
        for log in inspector_logs:
            log_lower = log.lower()
            if 'shape' in log_lower or f'{len(df_original)} rows' in log_lower:
                characteristics['shape_identified'] = True
            if 'dtype' in log_lower or 'type' in log_lower:
                characteristics['dtypes_identified'] = True
            if 'describe' in log_lower or 'mean' in log_lower or 'std' in log_lower:
                characteristics['statistics_computed'] = True
            if any(col in log for col in df_original.columns[:3]):  # Verifica primeiras 3 colunas
                characteristics['columns_listed'] = True
        
        analysis['characteristics_extracted'] = characteristics
        analysis['characteristics_score'] = sum(characteristics.values()) / len(characteristics) if characteristics else 0
        
        return analysis
    
    def analyze_imputator_node(self, logs: List[str], df_before: pd.DataFrame, 
                                df_after: pd.DataFrame, df_original_complete: pd.DataFrame = None) -> Dict:
        """
        Analisa o Nó Imputator
        Retorna:
        - Método escolhido (knn, mice, gp)
        - Valores inseridos
        - % erro em relação ao original (se disponível)
        """
        analysis = {
            'method_chosen': None,
            'parameters': {},
            'values_imputed_count': 0,
            'error_vs_original': None,
            'execution_success': False,
            'errors': []
        }
        
        # Procurar método escolhido nos logs
        imputator_logs = [log for log in logs if '[Imputator Node]' in log or 'imputat' in log.lower()]
        
        for log in imputator_logs:
            # Extrair método
            if 'method' in log.lower():
                for method in ['knn', 'mice', 'gp', 'gaussian', 'forward', 'backward', 'linear']:
                    if method in log.lower():
                        analysis['method_chosen'] = method
                        break
            
            # Extrair parâmetros
            if 'params' in log.lower() or 'parameters' in log.lower():
                # Tentar extrair JSON de parâmetros
                try:
                    json_match = re.search(r'\{.*?\}', log)
                    if json_match:
                        params = json.loads(json_match.group())
                        analysis['parameters'] = params
                except Exception:
                    pass
            
            # Verificar sucesso
            if 'success' in log.lower() or 'completed' in log.lower() or 'done' in log.lower():
                analysis['execution_success'] = True
            
            # Verificar erros
            if 'error' in log.lower() or 'failed' in log.lower():
                analysis['errors'].append(log)
        
        # Calcular valores imputados
        missing_before = df_before.isna().sum().sum()
        missing_after = df_after.isna().sum().sum()
        analysis['values_imputed_count'] = missing_before - missing_after
        
        # Se reduziu missing values, considerar sucesso
        if analysis['values_imputed_count'] > 0:
            analysis['execution_success'] = True
        
        # Se temos o dataset original completo, calcular erro
        if df_original_complete is not None and analysis['values_imputed_count'] > 0:
            # Identificar posições que foram imputadas
            mask_imputed = df_before.isna() & ~df_after.isna()
            
            # Calcular erro apenas nas posições imputadas
            errors = []
            for col in df_after.select_dtypes(include=[np.number]).columns:
                if col in df_original_complete.columns and col in mask_imputed.columns:
                    mask_col = mask_imputed[col]
                    if mask_col.sum() > 0:
                        try:
                            original_vals = df_original_complete.loc[mask_col, col].values
                            imputed_vals = df_after.loc[mask_col, col].values
                            
                            # RMSE
                            rmse = np.sqrt(np.mean((original_vals - imputed_vals) ** 2))
                            # MAE
                            mae = np.mean(np.abs(original_vals - imputed_vals))
                            # MAPE (evitar divisão por zero)
                            non_zero_mask = np.abs(original_vals) > 1e-10
                            if non_zero_mask.sum() > 0:
                                mape = np.mean(np.abs((original_vals[non_zero_mask] - imputed_vals[non_zero_mask]) / original_vals[non_zero_mask])) * 100
                            else:
                                mape = mae  # Usar MAE como fallback
                            
                            errors.append({
                                'column': col,
                                'rmse': rmse,
                                'mae': mae,
                                'mape': mape,
                                'n_imputed': mask_col.sum()
                            })
                        except Exception as e:
                            print(f"    ⚠️  Erro ao calcular métricas para {col}: {e}")
            
            if errors:
                analysis['error_vs_original'] = errors
                analysis['avg_mape'] = np.mean([e['mape'] for e in errors])
                analysis['avg_rmse'] = np.mean([e['rmse'] for e in errors])
                analysis['avg_mae'] = np.mean([e['mae'] for e in errors])
        
        return analysis
    
    def analyze_feature_engineer_node(self, logs: List[str], df_before: pd.DataFrame, 
                                       df_after: pd.DataFrame) -> Dict:
        """
        Analisa o Nó Feature Engineer
        Retorna:
        - Features adicionadas
        - Quais features
        - % faltante após engenharia
        """
        analysis = {
            'features_added': [],
            'features_count': 0,
            'missing_percentage_after': 0.0,
            'feature_types': {},
            'errors': []
        }
        
        # Identificar colunas adicionadas
        cols_before = set(df_before.columns) if df_before is not None else set()
        cols_after = set(df_after.columns) if df_after is not None else set()
        
        # Tentar normalizar índices diferentes
        try:
            if df_before is not None and df_after is not None:
                if isinstance(df_before.index, pd.DatetimeIndex) != isinstance(df_after.index, pd.DatetimeIndex):
                    # Um tem index datetime, outro não - tentar normalizar
                    if isinstance(df_after.index, pd.DatetimeIndex) and 'date' not in cols_after:
                        cols_after.add('date')  # date foi movido para index
        except Exception:
            pass
        
        new_columns = list(cols_after - cols_before)
        
        # Se não detectou features por diferença de colunas, tentar extrair dos logs
        if not new_columns:
            for log in logs:
                # Procurar por padrões de features criadas
                if 'ewma' in log.lower() or 'moving average' in log.lower():
                    ewma_patterns = re.findall(r'(\w+_ewma_\d+h?)', log)
                    new_columns.extend(ewma_patterns)
                if 'lag' in log.lower():
                    lag_patterns = re.findall(r'(\w+_lag_\d+h?)', log)
                    new_columns.extend(lag_patterns)
                if 'day_of_week' in log.lower() or 'month' in log.lower() or 'hour' in log.lower():
                    temporal_features = ['day_of_week', 'month', 'hour']
                    new_columns.extend([f for f in temporal_features if f in log.lower()])
            
            # Remover duplicatas
            new_columns = list(set(new_columns))
        
        analysis['features_added'] = new_columns
        analysis['features_count'] = len(new_columns)
        
        # Calcular % faltante após
        if df_after is not None and df_after.size > 0:
            total_cells = df_after.size
            missing_cells = df_after.isna().sum().sum()
            analysis['missing_percentage_after'] = (missing_cells / total_cells) * 100
        
        # Classificar tipos de features criadas
        feature_types = {
            'rolling_window': [],
            'lagged': [],
            'temporal': [],
            'interaction': [],
            'statistical': [],
            'other': []
        }
        
        for col in new_columns:
            col_lower = col.lower()
            if 'rolling' in col_lower or 'ma' in col_lower or 'ewm' in col_lower:
                feature_types['rolling_window'].append(col)
            elif 'lag' in col_lower or 'shift' in col_lower:
                feature_types['lagged'].append(col)
            elif any(t in col_lower for t in ['hour', 'day', 'month', 'year', 'weekday']):
                feature_types['temporal'].append(col)
            elif '_x_' in col_lower or '*' in col_lower or 'ratio' in col_lower:
                feature_types['interaction'].append(col)
            elif any(s in col_lower for s in ['mean', 'std', 'var', 'min', 'max']):
                feature_types['statistical'].append(col)
            else:
                feature_types['other'].append(col)
        
        # Remover tipos vazios
        analysis['feature_types'] = {k: v for k, v in feature_types.items() if v}
        
        # Procurar nos logs para validar
        fe_logs = [log for log in logs if 'FeatureEngineering' in log or 'feature' in log.lower()]
        
        for log in fe_logs:
            if 'error' in log.lower() and 'Error' in log:
                analysis['errors'].append(log)
        
        return analysis
    
    def analyze_plotter_node(self, logs: List[str], images_path: str) -> Dict:
        """
        Analisa o Nó Plotter
        Retorna:
        - Quantidade de plots gerados
        - Quais plots (tipos)
        """
        analysis = {
            'plots_generated': 0,
            'plot_types': [],
            'plot_files': [],
            'errors': []
        }
        
        # Verificar arquivos de plot gerados
        if os.path.exists(images_path):
            try:
                plot_files = [f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg', '.svg'))]
                analysis['plots_generated'] = len(plot_files)
                analysis['plot_files'] = plot_files
                
                # Inferir tipos de plots pelos nomes
                for fname in plot_files:
                    fname_lower = fname.lower()
                    if 'scatter' in fname_lower:
                        analysis['plot_types'].append('scatter')
                    elif 'hist' in fname_lower:
                        analysis['plot_types'].append('histogram')
                    elif 'heat' in fname_lower or 'corr' in fname_lower:
                        analysis['plot_types'].append('heatmap')
                    elif 'time' in fname_lower or 'series' in fname_lower:
                        analysis['plot_types'].append('timeseries')
                    elif 'box' in fname_lower:
                        analysis['plot_types'].append('boxplot')
                    elif 'dist' in fname_lower:
                        analysis['plot_types'].append('distribution')
                    else:
                        analysis['plot_types'].append('other')
            except Exception as e:
                analysis['errors'].append(f"Erro ao listar arquivos de imagem: {e}")
        
        # Procurar nos logs
        plotter_logs = [log for log in logs if '[Plotter Node]' in log or 'plot' in log.lower()]
        
        for log in plotter_logs:
            if 'error' in log.lower():
                analysis['errors'].append(log)
        
        return analysis
    
    def analyze_supervisor_node(self, logs: List[str]) -> Dict:
        """
        Analisa o Nó Supervisor
        Retorna:
        - Decisões tomadas
        - Sequência de nós ativados
        - Loops detectados
        - Erros (quantos e quantos foram resolvidos)
        """
        analysis = {
            'decisions': [],
            'sequence': [],
            'loops_detected': 0,
            'total_errors': 0,
            'errors_resolved': 0,
            'errors_unresolved': 0,
            'planning_quality': 0.0
        }
        
        # Extrair decisões do supervisor
        for i, log in enumerate(logs):
            if 'supervisor' in log.lower() or 'next' in log.lower():
                # Extrair próximo nó
                for node in ['inspect', 'imputator', 'feature_engineer', 'automl', 'plotter', 'END']:
                    if node in log:
                        decision = {
                            'step': len(analysis['decisions']) + 1,
                            'next_node': node,
                            'log_index': i
                        }
                        analysis['decisions'].append(decision)
                        analysis['sequence'].append(node)
                        break
        
        # Detectar loops (mesma sequência de nós repetida)
        # Tipo 1: Loops de repetição imediata (mesmo nó várias vezes seguidas)
        consecutive_repeats = 0
        for i in range(len(analysis['sequence']) - 1):
            if analysis['sequence'][i] == analysis['sequence'][i + 1]:
                consecutive_repeats += 1
        
        # Tipo 2: Loops de padrão (mesma sequência repetida)
        pattern_loops = 0
        if len(analysis['sequence']) > 3:
            for i in range(len(analysis['sequence']) - 3):
                subseq = tuple(analysis['sequence'][i:i+3])
                rest = analysis['sequence'][i+3:]
                if len(rest) >= 3:
                    for j in range(len(rest) - 2):
                        if tuple(rest[j:j+3]) == subseq:
                            pattern_loops += 1
        
        analysis['loops_detected'] = consecutive_repeats + pattern_loops
        
        # Detectar "stuck loops" - mais de 5 repetições do mesmo nó
        if consecutive_repeats > 5:
            analysis['stuck_node'] = analysis['sequence'][-1] if analysis['sequence'] else 'unknown'
        
        # Contar erros
        error_pattern = re.compile(r'\[.*\]\s*(error|failed|exception)', re.IGNORECASE)
        resolved_pattern = re.compile(r'(resolved|corrected|fixed|retry.*success)', re.IGNORECASE)
        
        error_indices = []
        for i, log in enumerate(logs):
            if error_pattern.search(log):
                analysis['total_errors'] += 1
                error_indices.append(i)
        
        # Verificar quais erros foram resolvidos (tentativa subsequente bem-sucedida)
        for err_idx in error_indices:
            # Procurar nos próximos 5 logs
            resolved = False
            for j in range(err_idx + 1, min(err_idx + 6, len(logs))):
                if resolved_pattern.search(logs[j]) or 'success' in logs[j].lower():
                    resolved = True
                    break
            
            if resolved:
                analysis['errors_resolved'] += 1
            else:
                analysis['errors_unresolved'] += 1
        
        # Calcular qualidade do planejamento
        # (baseado em: poucos loops, alta taxa de resolução de erros, sequência lógica)
        loop_penalty = min(analysis['loops_detected'] * 0.1, 0.5)
        error_resolution_rate = (analysis['errors_resolved'] / analysis['total_errors']) if analysis['total_errors'] > 0 else 1.0
        
        analysis['planning_quality'] = max(0, min(1, (1 - loop_penalty) * error_resolution_rate))
        
        return analysis
    
    def analyze_automl_node(self, logs: List[str]) -> Dict:
        """
        Analisa o Nó AutoML (se executado)
        """
        analysis = {
            'executed': False,
            'model_selected': None,
            'metrics': {},
            'errors': []
        }
        
        automl_logs = [log for log in logs if 'automl' in log.lower() or 'AutoML' in log]
        
        if not automl_logs:
            return analysis
        
        analysis['executed'] = True
        
        for log in automl_logs:
            # Extrair modelo selecionado
            if 'best' in log.lower() and 'model' in log.lower():
                # Tentar extrair nome do modelo
                models = ['NaiveMean', 'SeasonalNaive', 'ETS', 'ARIMA', 'Prophet', 'DeepAR', 'AutoGluon']
                for model in models:
                    if model in log:
                        analysis['model_selected'] = model
                        break
            
            # Extrair métricas
            if 'mape' in log.lower():
                mape_match = re.search(r'mape[:\s=]+(\d+\.?\d*)', log.lower())
                if mape_match:
                    analysis['metrics']['MAPE'] = float(mape_match.group(1))
            
            if 'rmse' in log.lower():
                rmse_match = re.search(r'rmse[:\s=]+(\d+\.?\d*)', log.lower())
                if rmse_match:
                    analysis['metrics']['RMSE'] = float(rmse_match.group(1))
            
            if 'mae' in log.lower():
                mae_match = re.search(r'mae[:\s=]+(\d+\.?\d*)', log.lower())
                if mae_match:
                    analysis['metrics']['MAE'] = float(mae_match.group(1))
            
            if 'error' in log.lower():
                analysis['errors'].append(log)
        
        return analysis


class EndToEndTester:
    """Classe principal para executar testes end-to-end"""
    
    def __init__(self, datasets_path: str, output_path: str = "test_results"):
        self.datasets_path = Path(datasets_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)
        
        # Procurar automaticamente por datasets CSV
        self.datasets = self._discover_datasets()
        
        self.analyzer = AgentNodeAnalyzer()
        self.results = {}
    
    def _discover_datasets(self) -> Dict[str, str]:
        """Descobre automaticamente datasets CSV no diretório"""
        datasets = {}
        
        if self.datasets_path.exists():
            for file_path in self.datasets_path.glob("*.csv"):
                dataset_name = file_path.stem
                datasets[dataset_name] = file_path.name
        
        return datasets
    
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Carrega dataset pelo nome"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' não encontrado")
        
        file_path = self.datasets_path / self.datasets[dataset_name]
        
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix == '.npz':
            # Carregar .npz e converter para DataFrame
            data = np.load(file_path)
            # Assumindo que tem chave 'data' ou similar
            if 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                # Pegar primeira chave disponível
                key = list(data.keys())[0]
                df = pd.DataFrame(data[key])
            return df
        else:
            raise ValueError(f"Formato não suportado: {file_path.suffix}")
    
    def create_missing_data(self, df: pd.DataFrame, missing_rate: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Cria dados faltantes artificialmente para teste
        Retorna: (df_com_missing, df_original)
        """
        df_original = df.copy()
        df_missing = df.copy()
        
        # Selecionar colunas numéricas
        numeric_cols = df_missing.select_dtypes(include=[np.number]).columns
        
        # Criar missing values aleatórios
        for col in numeric_cols:
            mask = np.random.random(len(df_missing)) < missing_rate
            df_missing.loc[mask, col] = np.nan
        
        return df_missing, df_original
    
    def run_agent_test(self, dataset_name: str, csv_path: str, 
                       prompt: str = "Perform a complete exploratory data analysis and handle missing values") -> Dict:
        """
        Executa o agente e captura resultados
        Esta função integra com o WorkflowExecutor real
        """
        print(f"  Executando agente para {dataset_name}...")
        
        # Importar módulos necessários
        try:
            from langchain_community.chat_models import ChatDeepInfra
            from agentai.workflow import WorkflowExecutor
            from uuid import uuid4
            from dotenv import load_dotenv
            
            # Carregar API key - buscar .env no projeto
            env_path = None
            current = Path.cwd()
            for _ in range(3):  # Buscar até 3 níveis acima
                test_path = current / ".env"
                if test_path.exists():
                    env_path = test_path
                    break
                current = current.parent
            
            if env_path:
                load_dotenv(dotenv_path=env_path)
            else:
                load_dotenv()
            
            api_key = os.getenv("DEEPINFRA_API_KEY")
            if not api_key:
                raise ValueError(f"DEEPINFRA_API_KEY não encontrada (procurou em: {env_path or 'padrão'})")
            
            # Criar LLM
            llm = ChatDeepInfra(model="Qwen/Qwen2.5-72B-Instruct", max_tokens=500)
            
            # Configurar paths
            images_path = self.output_path / dataset_name / "plots"
            images_path.mkdir(parents=True, exist_ok=True)
            
            # Criar executor
            executor = WorkflowExecutor(
                llm=llm,
                csv_path=csv_path,
                plot_images_path=str(images_path)
            )
            
            # Executar
            thread_id = str(uuid4())
            final_state = executor.invoke(
                initial_message=prompt,
                thread_id=thread_id
            )
            
            # Carregar DataFrame original
            df_original = pd.read_csv(csv_path)
            
            # Estrutura de retorno
            return {
                'logs': final_state.get('logs', []),
                'df_original': df_original,
                'df_after_inspection': executor.df.copy(),
                'df_after_imputation': executor.df.copy(),
                'df_after_feature_eng': executor.df.copy(),
                'df_final': executor.df.copy(),
                'summary': final_state.get('summary', ''),
                'images_path': str(images_path)
            }
            
        except Exception as e:
            print(f"    ❌ Erro ao executar agente: {e}")
            import traceback
            traceback.print_exc()
            
            # Retornar estrutura vazia em caso de erro
            return {
                'logs': [f"ERROR: {str(e)}"],
                'df_original': pd.DataFrame(),
                'df_after_inspection': pd.DataFrame(),
                'df_after_imputation': pd.DataFrame(),
                'df_after_feature_eng': pd.DataFrame(),
                'df_final': pd.DataFrame(),
                'summary': f"Erro durante execução: {str(e)}",
                'images_path': ""
            }
    
    def generate_report(self, dataset_name: str, analysis_results: Dict) -> str:
        """Gera relatório formatado para um dataset"""
        report = f"""
{'='*80}
RELATÓRIO DE TESTE END-TO-END: {dataset_name}
Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

## 1. NÓ INSPETOR
{'─'*80}
"""
        
        inspector = analysis_results['inspector']
        missing_pct = inspector.get('missing_percentage_identified')
        report += f"• % Faltantes Identificados: {missing_pct:.2f}%" if missing_pct is not None else "• % Faltantes Identificados: N/A"
        report += "\n\n• Características Extraídas:\n"
        
        if 'characteristics_extracted' in inspector:
            for char, identified in inspector['characteristics_extracted'].items():
                status = "✓" if identified else "✗"
                report += f"  {status} {char.replace('_', ' ').title()}\n"
        
        report += f"\n• Score de Extração: {inspector.get('characteristics_score', 0):.2%}\n"
        
        if inspector.get('errors'):
            report += "\n⚠️  ERROS:\n"
            for err in inspector['errors']:
                report += f"  - {err}\n"
        
        report += f"""
## 2. NÓ IMPUTATOR
{'─'*80}
• Método Escolhido: {analysis_results['imputator'].get('method_chosen', 'N/A')}
• Parâmetros: {analysis_results['imputator'].get('parameters', {})}
• Valores Imputados: {analysis_results['imputator'].get('values_imputed_count', 0)}
• Execução Bem-Sucedida: {'Sim' if analysis_results['imputator'].get('execution_success') else 'Não'}
"""
        
        if analysis_results['imputator'].get('error_vs_original'):
            report += "\n• Erro vs Original:\n"
            report += f"  {'Coluna':<20} {'MAPE (%)':<12} {'RMSE':<12} {'MAE':<12}\n"
            report += f"  {'-'*56}\n"
            for err_info in analysis_results['imputator']['error_vs_original']:
                report += f"  {err_info['column']:<20} "
                report += f"{err_info['mape']:>10.2f}% "
                report += f"{err_info['rmse']:>10.4f}  "
                report += f"{err_info['mae']:>10.4f}\n"
            if 'avg_mape' in analysis_results['imputator']:
                report += f"\n  MAPE Médio: {analysis_results['imputator']['avg_mape']:.2f}%\n"
        
        report += f"""
## 3. NÓ FEATURE ENGINEERING
{'─'*80}
• Features Adicionadas: {analysis_results['feature_engineer'].get('features_count', 0)}
• % Faltante Após: {analysis_results['feature_engineer'].get('missing_percentage_after', 0):.2f}%

• Tipos de Features Criadas:
"""
        
        if analysis_results['feature_engineer'].get('feature_types'):
            for ftype, features in analysis_results['feature_engineer']['feature_types'].items():
                report += f"  - {ftype.replace('_', ' ').title()} ({len(features)}): {', '.join(features[:3])}"
                if len(features) > 3:
                    report += f"... (+{len(features)-3} mais)"
                report += "\n"
        
        report += f"""
## 4. NÓ PLOTTER
{'─'*80}
• Plots Gerados: {analysis_results['plotter'].get('plots_generated', 0)}
• Tipos: {', '.join(set(analysis_results['plotter'].get('plot_types', [])))}
"""
        
        if analysis_results['plotter'].get('plot_files'):
            report += "• Arquivos:\n"
            for pfile in analysis_results['plotter']['plot_files'][:10]:
                report += f"  - {pfile}\n"
            if len(analysis_results['plotter']['plot_files']) > 10:
                report += f"  ... (+{len(analysis_results['plotter']['plot_files'])-10} mais)\n"
        
        report += f"""
## 5. NÓ SUPERVISOR
{'─'*80}
• Decisões Tomadas: {len(analysis_results['supervisor'].get('decisions', []))}
"""
        
        if analysis_results['supervisor'].get('sequence'):
            report += f"• Sequência de Nós: {' → '.join(analysis_results['supervisor']['sequence'])}\n"
        
        report += f"• Loops Detectados: {analysis_results['supervisor'].get('loops_detected', 0)}\n"
        
        # Adicionar aviso se travou em loop
        if 'stuck_node' in analysis_results['supervisor']:
            report += f"  ⚠️ ATENÇÃO: Travou em loop no nó '{analysis_results['supervisor']['stuck_node']}'\n"
            report += f"  💡 Sugestão: Verifique o código do nó para corrigir o erro que causa o loop.\n"
        
        report += f"""• Total de Erros: {analysis_results['supervisor'].get('total_errors', 0)}
  - Resolvidos: {analysis_results['supervisor'].get('errors_resolved', 0)}
  - Não Resolvidos: {analysis_results['supervisor'].get('errors_unresolved', 0)}
• Qualidade do Planejamento: {analysis_results['supervisor'].get('planning_quality', 0):.2%}
"""
        
        if analysis_results.get('automl', {}).get('executed'):
            report += f"""
## 6. NÓ AUTOML
{'─'*80}
• Executado: Sim
• Modelo Selecionado: {analysis_results['automl'].get('model_selected', 'N/A')}
"""
            if analysis_results['automl'].get('metrics'):
                report += "• Métricas:\n"
                for metric, value in analysis_results['automl']['metrics'].items():
                    report += f"  - {metric}: {value:.4f}\n"
        
        report += f"\n{'='*80}\n"
        
        return report
    
    def generate_comparative_summary(self) -> str:
        """Gera relatório comparativo entre todos os datasets"""
        summary = f"""
{'='*80}
RESUMO COMPARATIVO - TODOS OS DATASETS
Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

{'Dataset':<15} {'Inspector':<12} {'Imputator':<12} {'Feat.Eng':<12} {'Plotter':<12} {'Supervisor':<12} {'Overall':<12}
{'-'*90}
"""
        
        for dataset_name, results in self.results.items():
            inspector_score = results['inspector'].get('characteristics_score', 0)
            imputator_success = 1.0 if results['imputator'].get('execution_success') else 0.0
            feat_eng_score = min(1.0, results['feature_engineer'].get('features_count', 0) / 10)
            plotter_score = min(1.0, results['plotter'].get('plots_generated', 0) / 4)
            supervisor_score = results['supervisor'].get('planning_quality', 0)
            
            overall = np.mean([inspector_score, imputator_success, feat_eng_score, 
                              plotter_score, supervisor_score])
            
            summary += f"{dataset_name:<15} "
            summary += f"{inspector_score:>10.1%} "
            summary += f"{imputator_success:>11.1%} "
            summary += f"{feat_eng_score:>11.1%} "
            summary += f"{plotter_score:>11.1%} "
            summary += f"{supervisor_score:>11.1%} "
            summary += f"{overall:>11.1%}\n"
        
        summary += f"\n{'='*80}\n"
        
        return summary
    
    def run_all_tests(self, prompt: str = None):
        """Executa testes para todos os datasets"""
        
        if prompt is None:
            prompt = """
Perform a complete exploratory data analysis on this time series dataset:
1. Inspect the data structure, types, and identify missing values
2. Apply appropriate imputation techniques for missing data
3. Create relevant time series features (lags, rolling windows, etc.)
4. Generate visualizations (time series plots, correlations, distributions)
5. Provide a comprehensive summary of findings
"""
        
        print(f"\n{'='*80}")
        print("INICIANDO TESTES END-TO-END")
        print(f"{'='*80}\n")
        
        if not self.datasets:
            print("❌ Nenhum dataset encontrado!")
            return
        
        print(f"Datasets encontrados: {', '.join(self.datasets.keys())}\n")
        
        for dataset_name in self.datasets.keys():
            print(f"\n{'─'*80}")
            print(f"Testando: {dataset_name}")
            print(f"{'─'*80}\n")
            
            try:
                # Carregar dataset
                df = self.load_dataset(dataset_name)
                print(f"✓ Dataset carregado: {df.shape}")
                
                # Criar missing data para teste
                df_missing, df_original = self.create_missing_data(df, missing_rate=0.15)
                
                # Salvar CSV temporário
                temp_csv = self.output_path / f"{dataset_name}_test.csv"
                df_missing.to_csv(temp_csv, index=False)
                print(f"✓ Dataset com missing salvo: {temp_csv}")
                
                # Executar agente
                agent_results = self.run_agent_test(dataset_name, str(temp_csv), prompt)
                
                # Analisar cada nó
                analysis = {}
                
                print("\nAnalisando resultados...")
                analysis['inspector'] = self.analyzer.analyze_inspector_node(
                    agent_results['logs'],
                    df_missing,
                    agent_results.get('df_after_inspection', df_missing)
                )
                
                analysis['imputator'] = self.analyzer.analyze_imputator_node(
                    agent_results['logs'],
                    agent_results.get('df_after_inspection', df_missing),
                    agent_results.get('df_after_imputation', df_missing),
                    df_original
                )
                
                analysis['feature_engineer'] = self.analyzer.analyze_feature_engineer_node(
                    agent_results['logs'],
                    agent_results.get('df_after_imputation', df_missing),
                    agent_results.get('df_after_feature_eng', df_missing)
                )
                
                analysis['plotter'] = self.analyzer.analyze_plotter_node(
                    agent_results['logs'],
                    agent_results['images_path']
                )
                
                analysis['supervisor'] = self.analyzer.analyze_supervisor_node(
                    agent_results['logs']
                )
                
                analysis['automl'] = self.analyzer.analyze_automl_node(
                    agent_results['logs']
                )
                
                self.results[dataset_name] = analysis
                
                # Gerar relatório individual
                report = self.generate_report(dataset_name, analysis)
                
                # Salvar relatório
                report_path = self.output_path / f"{dataset_name}_report.txt"
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                print(f"\n✓ Relatório salvo em: {report_path}")
                
            except Exception as e:
                print(f"✗ Erro ao testar {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Gerar resumo comparativo
        if self.results:
            comparative_summary = self.generate_comparative_summary()
            print(f"\n{comparative_summary}")
            
            summary_path = self.output_path / "comparative_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(comparative_summary)
            
            print(f"✓ Resumo comparativo salvo em: {summary_path}")
            
            # Gerar visualizações
            self.generate_visualizations()
    
    def generate_visualizations(self):
        """Gera gráficos comparativos dos resultados"""
        if not self.results:
            return
        
        try:
            # Preparar dados para visualização
            datasets = list(self.results.keys())
            
            metrics = {
                'Inspector': [],
                'Imputator': [],
                'Feature Eng': [],
                'Plotter': [],
                'Supervisor': []
            }
            
            for dataset_name in datasets:
                results = self.results[dataset_name]
                metrics['Inspector'].append(results['inspector'].get('characteristics_score', 0))
                metrics['Imputator'].append(1.0 if results['imputator'].get('execution_success') else 0.0)
                metrics['Feature Eng'].append(min(1.0, results['feature_engineer'].get('features_count', 0) / 10))
                metrics['Plotter'].append(min(1.0, results['plotter'].get('plots_generated', 0) / 4))
                metrics['Supervisor'].append(results['supervisor'].get('planning_quality', 0))
            
            # Criar gráfico de barras
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Gráfico 1: Desempenho por nó
            x = np.arange(len(datasets))
            width = 0.15
            
            for i, (node, scores) in enumerate(metrics.items()):
                ax1.bar(x + i * width, scores, width, label=node)
            
            ax1.set_xlabel('Datasets')
            ax1.set_ylabel('Score (0-1)')
            ax1.set_title('Desempenho por Nó e Dataset')
            ax1.set_xticks(x + width * 2)
            ax1.set_xticklabels(datasets, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            
            # Gráfico 2: Heatmap de desempenho
            data_matrix = np.array([metrics[node] for node in metrics.keys()])
            im = ax2.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            
            ax2.set_xticks(np.arange(len(datasets)))
            ax2.set_yticks(np.arange(len(metrics)))
            ax2.set_xticklabels(datasets, rotation=45, ha='right')
            ax2.set_yticklabels(metrics.keys())
            ax2.set_title('Heatmap de Desempenho')
            
            # Adicionar valores no heatmap
            for i in range(len(metrics)):
                for j in range(len(datasets)):
                    text = ax2.text(j, i, f'{data_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=9)
            
            plt.colorbar(im, ax=ax2, label='Score')
            
            plt.tight_layout()
            plot_path = self.output_path / "performance_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Visualização salva em: {plot_path}")
        
        except Exception as e:
            print(f"⚠️  Erro ao gerar visualizações: {e}")


def main():
    """Função principal para executar os testes"""
    
    # Detectar diretório do script e projeto
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent if script_dir.name == 'tests' else script_dir
    
    # Configurar caminhos relativos ao projeto
    datasets_path = project_root / "agentai" / "datasets"
    output_path = project_root / "test_results_e2e"
    
    # Criar tester
    print("Inicializando framework de testes...")
    print(f"Projeto: {project_root}")
    print(f"Datasets: {datasets_path}")
    print(f"Output: {output_path}\n")
    
    tester = EndToEndTester(str(datasets_path), str(output_path))
    
    # Prompt de teste padrão
    test_prompt = """
Perform a complete exploratory data analysis on this dataset:
1. Inspect the data structure, types, and identify missing values
2. Apply appropriate imputation techniques for missing data
3. Create relevant time series features (lags, rolling windows, etc.)
4. Generate visualizations (time series plots, correlations, distributions)
5. Provide a comprehensive summary of findings
"""
    
    # Executar todos os testes
    tester.run_all_tests(prompt=test_prompt)
    
    print(f"\n{'='*80}")
    print("TESTES CONCLUÍDOS")
    print(f"Resultados salvos em: {Path(output_path).absolute()}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()