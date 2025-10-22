import pandas as pd
import numpy as np
import time
import json
import os
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Adiciona o diretório do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agentai.workflow import WorkflowExecutor
from agentai.nodes import FeatureEngineeringNode
from agentai.modules.common import AgentState
from langchain_community.chat_models import ChatDeepInfra
from dotenv import load_dotenv


class FeatureEngDatasetGenerator:
    """Gera datasets específicos para testar feature engineering"""
    
    @staticmethod
    def generate_timeseries_iot(
        n_rows: int = 500,
        freq: str = '1H',
        seasonality: bool = True,
        trend: bool = True,
        noise_level: float = 0.5,
        missing_rate: float = 0.0
    ) -> pd.DataFrame:
        """
        Gera série temporal simulando dados de sensores IoT
        
        Args:
            n_rows: número de observações
            freq: frequência temporal ('1H', '15min', etc)
            seasonality: incluir padrão sazonal
            trend: incluir tendência
            noise_level: nível de ruído (0 a 1)
            missing_rate: taxa de dados faltantes
        """
        print(f"   Gerando IoT dataset: {n_rows} pontos, freq={freq}, "
              f"sazonalidade={seasonality}, tendência={trend}")
        
        # Criar índice temporal
        start_date = datetime(2024, 1, 1)
        date_range = pd.date_range(start=start_date, periods=n_rows, freq=freq)
        
        # Simular temperatura
        temperature = np.zeros(n_rows)
        if trend:
            temperature += np.linspace(20, 25, n_rows)  # Tendência de aquecimento
        if seasonality:
            # Ciclo diário
            hours = np.arange(n_rows) % 24
            temperature += 5 * np.sin(2 * np.pi * hours / 24)
        temperature += np.random.normal(0, noise_level, n_rows)
        
        # Simular umidade (correlacionada inversamente com temperatura)
        humidity = 100 - temperature * 2 + np.random.normal(0, noise_level * 2, n_rows)
        humidity = np.clip(humidity, 0, 100)
        
        # Simular pressão (mais estável)
        pressure = 1013 + np.random.normal(0, 2, n_rows)
        
        df = pd.DataFrame({
            'timestamp': date_range,
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure
        })
        
        # Adicionar dados faltantes
        if missing_rate > 0:
            n_missing = int(n_rows * missing_rate)
            for col in ['temperature', 'humidity', 'pressure']:
                missing_idx = np.random.choice(n_rows, size=n_missing, replace=False)
                df.loc[missing_idx, col] = np.nan
        
        return df
    
    @staticmethod
    def generate_multivariate_iot(
        n_rows: int = 500,
        n_sensors: int = 5
    ) -> pd.DataFrame:
        """Gera dataset com múltiplos sensores para testar features de interação"""
        print(f"   Gerando dataset multivariado: {n_rows} pontos, {n_sensors} sensores")
        
        start_date = datetime(2024, 1, 1)
        date_range = pd.date_range(start=start_date, periods=n_rows, freq='1H')
        
        data = {'timestamp': date_range}
        
        # Criar sensores correlacionados
        base_signal = np.linspace(20, 30, n_rows) + 5 * np.sin(np.linspace(0, 4*np.pi, n_rows))
        
        for i in range(n_sensors):
            # Cada sensor tem uma versão ligeiramente diferente do sinal base
            sensor_data = base_signal + np.random.normal(0, 1 + i*0.5, n_rows)
            data[f'sensor_{i}'] = sensor_data
        
        return pd.DataFrame(data)



class FeatureValidator:
    """Valida se as features foram criadas corretamente"""
    
    @staticmethod
    def validate_rolling_average(df: pd.DataFrame, original_cols: List[str], 
                                 window: int, target_col: str) -> Dict:
        """Valida se uma média móvel foi calculada corretamente"""
        new_cols = [col for col in df.columns if col not in original_cols]
        
        if len(new_cols) == 0:
            return {"success": False, "reason": "No new columns created"}
        
        # Verificar se pelo menos uma coluna parece ser uma rolling average
        for new_col in new_cols:
            if not df[new_col].isna().all():
                # Calcular rolling average esperado
                expected = df[target_col].rolling(window=window).mean()
                
                # Verificar correlação (deve ser alta)
                if not df[new_col].isna().all() and not expected.isna().all():
                    correlation = df[new_col].corr(expected)
                    if correlation > 0.95:  # Alta correlação indica sucesso
                        return {
                            "success": True,
                            "new_columns": new_cols,
                            "correlation": correlation,
                            "reason": f"Rolling average created correctly (corr={correlation:.3f})"
                        }
        
        return {
            "success": False,
            "new_columns": new_cols,
            "reason": "New column created but doesn't match expected rolling average"
        }
    
    @staticmethod
    def validate_lag_feature(df: pd.DataFrame, original_cols: List[str], 
                            lag: int, target_col: str) -> Dict:
        """Valida se um lag foi criado corretamente"""
        new_cols = [col for col in df.columns if col not in original_cols]
        
        if len(new_cols) == 0:
            return {"success": False, "reason": "No new columns created"}
        
        for new_col in new_cols:
            if not df[new_col].isna().all():
                # Verificar se é um lag correto
                expected = df[target_col].shift(lag)
                
                if not df[new_col].isna().all() and not expected.isna().all():
                    # Verificar se os valores são idênticos (exceto NaN)
                    valid_indices = ~(df[new_col].isna() | expected.isna())
                    if valid_indices.sum() > 0:
                        match_rate = (df.loc[valid_indices, new_col] == expected[valid_indices]).mean()
                        if match_rate > 0.95:
                            return {
                                "success": True,
                                "new_columns": new_cols,
                                "match_rate": match_rate,
                                "reason": f"Lag feature created correctly (match={match_rate:.3f})"
                            }
        
        return {
            "success": False,
            "new_columns": new_cols,
            "reason": "New column created but doesn't match expected lag"
        }
    
    @staticmethod
    def validate_new_columns(df: pd.DataFrame, original_cols: List[str], 
                           expected_min: int = 1) -> Dict:
        """Valida genericamente se novas colunas foram criadas"""
        new_cols = [col for col in df.columns if col not in original_cols]
        
        if len(new_cols) >= expected_min:
            return {
                "success": True,
                "new_columns": new_cols,
                "count": len(new_cols),
                "reason": f"{len(new_cols)} new column(s) created"
            }
        else:
            return {
                "success": False,
                "new_columns": new_cols,
                "count": len(new_cols),
                "reason": f"Expected at least {expected_min} columns, got {len(new_cols)}"
            }


@dataclass
class FeatureTestResult:
    """Armazena resultado de um teste de feature engineering"""
    test_name: str
    test_category: str
    success: bool
    execution_time: float
    dataset_size: int
    columns_before: int
    columns_after: int
    new_columns: List[str]
    validation_result: Dict
    error_message: str
    agent_output: str

class FeatureEngineeringTester:
    """Classe para testar o nó Feature Engineering"""
    
    def __init__(self):
        """Inicializa o testador"""
        print("\n" + "="*80)
        print("INICIALIZANDO TESTADOR DO NÓ FEATURE ENGINEERING")
        print("="*80)
        
        # Carregar variáveis de ambiente
        load_dotenv()
        
        # Inicializar LLM
        print("\n[1/3] Inicializando LLM...")
        self.llm = ChatDeepInfra(model="Qwen/Qwen2.5-72B-Instruct")
        print("   ✓ LLM inicializado")
        
        # Criar dataset dummy inicial
        print("\n[2/3] Criando dataset inicial...")
        dummy_df = FeatureEngDatasetGenerator.generate_timeseries_iot(n_rows=10)
        dummy_path = "temp_dummy_fe_dataset.csv"
        dummy_df.to_csv(dummy_path, index=False)
        print(f"   ✓ Dataset salvo em {dummy_path}")
        
        # Inicializar executor
        print("\n[3/3] Inicializando WorkflowExecutor...")
        self.executor = WorkflowExecutor(
            csv_path=dummy_path,
            plot_images_path="./test_plots",
            llm=self.llm
        )
        print("   ✓ Executor inicializado")
        
        self.results: List[FeatureTestResult] = []
        self.validator = FeatureValidator()
        
        print("\n✓ Testador pronto para executar testes!")
    
    def run_all_tests(self):
        """Executa todos os testes"""
        print("\n" + "="*80)
        print("EXECUTANDO BATERIA DE TESTES - FEATURE ENGINEERING")
        print("="*80)
        
        # CATEGORIA 1: Features Temporais (Rolling)
        print("\n[CATEGORIA 1/5] Features Temporais - Rolling Averages")
        print("-" * 80)
        self._test_simple_rolling_average()
        self._test_multiple_rolling_windows()
        self._test_rolling_with_missing_data()
        
        # CATEGORIA 2: Features Temporais (Lag)
        print("\n[CATEGORIA 2/5] Features Temporais - Lags")
        print("-" * 80)
        self._test_simple_lag()
        self._test_multiple_lags()
        
        # CATEGORIA 3: Features Estatísticas
        print("\n[CATEGORIA 3/5] Features Estatísticas")
        print("-" * 80)
        self._test_statistical_features()
        self._test_rolling_std()
        
        # CATEGORIA 4: Features de Interação
        print("\n[CATEGORIA 4/5] Features de Interação")
        print("-" * 80)
        self._test_ratio_features()
        self._test_difference_features()
        
        # CATEGORIA 5: Robustez e Escalabilidade
        print("\n[CATEGORIA 5/5] Robustez e Escalabilidade")
        print("-" * 80)
        self._test_large_dataset()
        self._test_multiple_features_at_once()
        
        print("\n" + "="*80)
        print("TESTES CONCLUÍDOS")
        print("="*80)
    
    def _run_single_test(
        self, 
        test_name: str, 
        test_category: str,
        df: pd.DataFrame, 
        instruction: str,
        validator_func=None,
        validator_args: Dict = None
    ):
        """Executa um único teste de feature engineering"""
        print(f"\n   [{test_category}] {test_name}")
        print(f"   Dataset: {len(df)} linhas, {len(df.columns)} colunas")
        print(f"   Instrução: {instruction[:70]}...")
        
        # Guardar estado original
        original_columns = df.columns.tolist()
        original_shape = df.shape
        
        # Substituir DataFrame no executor
        self.executor.df = df.copy()
        
        # Executar teste
        start_time = time.time()
        success = False
        error_message = ""
        agent_output = ""
        new_columns = []
        validation_result = {}
        
        try:
            # Criar nó e estado
            node = FeatureEngineeringNode(self.executor)
            state = AgentState(msg=instruction, logs=[])
            
            # Executar
            result = node.execute(state)
            execution_time = time.time() - start_time
            
            # Extrair output
            agent_output = result.get("subagents_report", "")
            
            # Pegar DataFrame modificado
            df_after = self.executor.df
            
            # Identificar novas colunas
            new_columns = [col for col in df_after.columns if col not in original_columns]
            
            # Validar resultado
            if validator_func:
                validation_result = validator_func(
                    df_after, 
                    original_columns, 
                    **(validator_args or {})
                )
                success = validation_result.get("success", False)
            else:
                # Validação genérica: pelo menos uma coluna foi criada
                validation_result = self.validator.validate_new_columns(
                    df_after, original_columns
                )
                success = validation_result.get("success", False)
            
            if not success:
                error_message = validation_result.get("reason", "Validation failed")
            
            status = "✓ SUCESSO" if success else "✗ FALHA"
            print(f"   {status} - Tempo: {execution_time:.2f}s - Novas colunas: {len(new_columns)}")
            if new_columns:
                print(f"   Colunas criadas: {new_columns}")
            
        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            error_message = str(e)
            validation_result = {"success": False, "reason": error_message}
            print(f"   ✗ ERRO - {error_message}")
        
        # Salvar resultado
        result = FeatureTestResult(
            test_name=test_name,
            test_category=test_category,
            success=success,
            execution_time=execution_time,
            dataset_size=original_shape[0],
            columns_before=original_shape[1],
            columns_after=len(df_after.columns) if 'df_after' in locals() else original_shape[1],
            new_columns=new_columns,
            validation_result=validation_result,
            error_message=error_message,
            agent_output=agent_output[:500]
        )
        self.results.append(result)
    
    
    def _test_simple_rolling_average(self):
        """Teste básico: criar média móvel de 3 períodos"""
        df = FeatureEngDatasetGenerator.generate_timeseries_iot(n_rows=200)
        self._run_single_test(
            test_name="rolling_avg_3h",
            test_category="Rolling",
            df=df,
            instruction="Create a 3-hour rolling average for the temperature column. Name it 'temp_rolling_3h'.",
            validator_func=self.validator.validate_rolling_average,
            validator_args={"window": 3, "target_col": "temperature"}
        )
    
    def _test_multiple_rolling_windows(self):
        """Teste: criar múltiplas janelas de rolling average"""
        df = FeatureEngDatasetGenerator.generate_timeseries_iot(n_rows=200)
        self._run_single_test(
            test_name="rolling_avg_multiple_windows",
            test_category="Rolling",
            df=df,
            instruction="Create rolling averages for temperature with windows of 3, 6, and 12 hours.",
            validator_func=self.validator.validate_new_columns,
            validator_args={"expected_min": 3}
        )
    
    def _test_rolling_with_missing_data(self):
        """Teste: rolling average com dados faltantes"""
        df = FeatureEngDatasetGenerator.generate_timeseries_iot(
            n_rows=200, missing_rate=0.1
        )
        self._run_single_test(
            test_name="rolling_avg_with_missing",
            test_category="Rolling",
            df=df,
            instruction="Create a 5-hour rolling average for temperature, handling missing values appropriately.",
            validator_func=self.validator.validate_rolling_average,
            validator_args={"window": 5, "target_col": "temperature"}
        )
    
    
    def _test_simple_lag(self):
        """Teste básico: criar lag de 1 período"""
        df = FeatureEngDatasetGenerator.generate_timeseries_iot(n_rows=200)
        self._run_single_test(
            test_name="lag_1h",
            test_category="Lag",
            df=df,
            instruction="Create a lag feature for temperature with lag=1 (previous hour value).",
            validator_func=self.validator.validate_lag_feature,
            validator_args={"lag": 1, "target_col": "temperature"}
        )
    
    def _test_multiple_lags(self):
        """Teste: criar múltiplos lags"""
        df = FeatureEngDatasetGenerator.generate_timeseries_iot(n_rows=200)
        self._run_single_test(
            test_name="lag_multiple",
            test_category="Lag",
            df=df,
            instruction="Create lag features for temperature with lags of 1, 3, and 6 hours.",
            validator_func=self.validator.validate_new_columns,
            validator_args={"expected_min": 3}
        )
    
    
    def _test_statistical_features(self):
        """Teste: criar features estatísticas (min, max, mean em janela)"""
        df = FeatureEngDatasetGenerator.generate_timeseries_iot(n_rows=200)
        self._run_single_test(
            test_name="statistical_window_features",
            test_category="Statistical",
            df=df,
            instruction="Create rolling min, max, and mean for temperature with a 6-hour window.",
            validator_func=self.validator.validate_new_columns,
            validator_args={"expected_min": 3}
        )
    
    def _test_rolling_std(self):
        """Teste: desvio padrão móvel"""
        df = FeatureEngDatasetGenerator.generate_timeseries_iot(n_rows=200)
        self._run_single_test(
            test_name="rolling_std",
            test_category="Statistical",
            df=df,
            instruction="Create a rolling standard deviation for temperature with a 12-hour window to measure variability.",
            validator_func=self.validator.validate_new_columns,
            validator_args={"expected_min": 1}
        )
    
    
    def _test_ratio_features(self):
        """Teste: criar ratios entre variáveis"""
        df = FeatureEngDatasetGenerator.generate_timeseries_iot(n_rows=200)
        self._run_single_test(
            test_name="ratio_features",
            test_category="Interaction",
            df=df,
            instruction="Create a ratio feature: temperature divided by humidity.",
            validator_func=self.validator.validate_new_columns,
            validator_args={"expected_min": 1}
        )
    
    def _test_difference_features(self):
        """Teste: criar diferenças temporais"""
        df = FeatureEngDatasetGenerator.generate_timeseries_iot(n_rows=200)
        self._run_single_test(
            test_name="difference_features",
            test_category="Interaction",
            df=df,
            instruction="Create a feature that represents the change in temperature from the previous hour (temperature difference).",
            validator_func=self.validator.validate_new_columns,
            validator_args={"expected_min": 1}
        )
    
    
    def _test_large_dataset(self):
        """Teste de escalabilidade com dataset grande"""
        df = FeatureEngDatasetGenerator.generate_timeseries_iot(n_rows=2000)
        self._run_single_test(
            test_name="large_dataset_2000_rows",
            test_category="Scalability",
            df=df,
            instruction="Create a 24-hour rolling average for temperature.",
            validator_func=self.validator.validate_rolling_average,
            validator_args={"window": 24, "target_col": "temperature"}
        )
    
    def _test_multiple_features_at_once(self):
        """Teste: criar múltiplas features complexas de uma vez"""
        df = FeatureEngDatasetGenerator.generate_multivariate_iot(n_rows=200, n_sensors=3)
        self._run_single_test(
            test_name="multiple_complex_features",
            test_category="Scalability",
            df=df,
            instruction="Create the following features: (1) 6-hour rolling average for sensor_0, (2) lag-1 for sensor_1, (3) ratio of sensor_0 to sensor_1.",
            validator_func=self.validator.validate_new_columns,
            validator_args={"expected_min": 3}
        )
    
    
    def generate_report(self):
        """Gera relatório dos resultados"""
        print("\n" + "="*80)
        print("RELATÓRIO DE RESULTADOS - FEATURE ENGINEERING")
        print("="*80)
        
        total = len(self.results)
        success = sum(1 for r in self.results if r.success)
        fail = total - success
        
        success_rate = (success / total * 100) if total > 0 else 0
        avg_time = np.mean([r.execution_time for r in self.results])
        
        # Análise por categoria
        categories = {}
        for r in self.results:
            cat = r.test_category
            if cat not in categories:
                categories[cat] = {"total": 0, "success": 0}
            categories[cat]["total"] += 1
            if r.success:
                categories[cat]["success"] += 1
        
        print(f"\n📊 RESUMO GERAL:")
        print(f"   Total de testes: {total}")
        print(f"   Sucessos: {success} ({success_rate:.1f}%)")
        print(f"   Falhas: {fail} ({100-success_rate:.1f}%)")
        print(f"   Tempo médio: {avg_time:.2f}s")
        
        print(f"\n📊 RESUMO POR CATEGORIA:")
        for cat, stats in categories.items():
            cat_rate = (stats["success"] / stats["total"] * 100)
            print(f"   {cat:15s}: {stats['success']}/{stats['total']} ({cat_rate:.1f}%)")
        
        print(f"\n📋 DETALHES POR TESTE:")
        print("-" * 80)
        for r in self.results:
            status = "✓" if r.success else "✗"
            print(f"{status} [{r.test_category:12s}] {r.test_name:35s} | "
                  f"Size: {r.dataset_size:4d} | "
                  f"Cols: {r.columns_before}→{r.columns_after} | "
                  f"Time: {r.execution_time:5.2f}s")
            if not r.success:
                print(f"   └─ Erro: {r.error_message[:70]}")
            elif r.new_columns:
                print(f"   └─ Criadas: {', '.join(r.new_columns[:3])}")
        
        return {
            "summary": {
                "total_tests": total,
                "successful": success,
                "failed": fail,
                "success_rate": f"{success_rate:.1f}%",
                "average_time": f"{avg_time:.2f}s"
            },
            "by_category": {
                cat: {
                    "total": stats["total"],
                    "success": stats["success"],
                    "rate": f"{stats['success']/stats['total']*100:.1f}%"
                }
                for cat, stats in categories.items()
            },
            "details": [asdict(r) for r in self.results]
        }
    
    def save_report(self, filename="feature_eng_evaluation_report.json"):
        """Salva relatório em arquivo JSON"""
        report = self.generate_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Relatório salvo em: {filename}")
    
    def plot_results(self, filename="feature_eng_evaluation_plots.png"):
        """Gera gráficos dos resultados"""
        print(f"\n📊 Gerando gráficos...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gráfico 1: Taxa de sucesso geral
        success_count = sum(1 for r in self.results if r.success)
        fail_count = len(self.results) - success_count
        
        axes[0, 0].bar(['Sucesso', 'Falha'], [success_count, fail_count], 
                      color=['green', 'red'], alpha=0.7)
        axes[0, 0].set_ylabel('Número de Testes')
        axes[0, 0].set_title('Resultado Geral dos Testes')
        axes[0, 0].set_ylim([0, len(self.results) + 1])
        
        for i, v in enumerate([success_count, fail_count]):
            axes[0, 0].text(i, v + 0.1, str(v), ha='center', fontweight='bold')
        
        # Gráfico 2: Taxa de sucesso por categoria
        categories = {}
        for r in self.results:
            cat = r.test_category
            if cat not in categories:
                categories[cat] = {"total": 0, "success": 0}
            categories[cat]["total"] += 1
            if r.success:
                categories[cat]["success"] += 1
        
        cat_names = list(categories.keys())
        cat_rates = [categories[c]["success"]/categories[c]["total"]*100 for c in cat_names]
        
        axes[0, 1].bar(cat_names, cat_rates, color='steelblue', alpha=0.7)
        axes[0, 1].set_ylabel('Taxa de Sucesso (%)')
        axes[0, 1].set_title('Taxa de Sucesso por Categoria')
        axes[0, 1].set_ylim([0, 105])
        axes[0, 1].axhline(y=80, color='r', linestyle='--', label='Meta: 80%')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Gráfico 3: Tempo de execução por categoria
        cat_times = {}
        for r in self.results:
            cat = r.test_category
            if cat not in cat_times:
                cat_times[cat] = []
            cat_times[cat].append(r.execution_time)
        
        cat_names_time = list(cat_times.keys())
        cat_avg_times = [np.mean(cat_times[c]) for c in cat_names_time]
        
        axes[1, 0].bar(cat_names_time, cat_avg_times, color='coral', alpha=0.7)
        axes[1, 0].set_ylabel('Tempo Médio (s)')
        axes[1, 0].set_title('Tempo Médio de Execução por Categoria')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Gráfico 4: Número de colunas criadas
        test_names = [r.test_name[:20] for r in self.results]
        cols_created = [len(r.new_columns) for r in self.results]
        colors = ['green' if r.success else 'red' for r in self.results]
        
        axes[1, 1].barh(test_names, cols_created, color=colors, alpha=0.7)
        axes[1, 1].set_xlabel('Número de Colunas Criadas')
        axes[1, 1].set_title('Colunas Criadas por Teste')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Gráficos salvos em: {filename}")
        plt.close()


def main():
    """Função principal para executar os testes"""
    
    print("\n" + "="*80)
    print(" SISTEMA DE TESTES DO NÓ FEATURE ENGINEERING ".center(80, "="))
    print("="*80)
    
    try:
        # Inicializar testador
        tester = FeatureEngineeringTester()
        
        # Executar testes
        tester.run_all_tests()
        
        # Gerar relatório
        tester.generate_report()
        
        # Salvar resultados
        tester.save_report("feature_eng_evaluation_report.json")
        tester.plot_results("feature_eng_evaluation_plots.png")
        
        print("\n" + "="*80)
        print(" TESTES CONCLUÍDOS COM SUCESSO ".center(80, "="))
        print("="*80)
        print("\n📁 Arquivos gerados:")
        print("   - feature_eng_evaluation_report.json (relatório detalhado)")
        print("   - feature_eng_evaluation_plots.png (gráficos)")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()