import pandas as pd
import numpy as np
import time
import json
import os
import sys
from typing import Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Adiciona o diretório do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agentai.workflow import WorkflowExecutor
from agentai.nodes import ImputatorNode
from agentai.modules.common import AgentState
from langchain_community.chat_models import ChatDeepInfra
from dotenv import load_dotenv


class ImputatorDatasetGenerator:
    """Gera datasets específicos para testar imputação"""
    
    @staticmethod
    def generate_sensor_data_missing(
        n_rows: int = 200,
        missing_rate: float = 0.2,
        pattern: str = "random"
    ) -> pd.DataFrame:
        """
        Gera dados de sensor com missing values
        
        Args:
            pattern: 'random', 'temporal' (blocos), 'column' (coluna inteira)
        """
        print(f"   Gerando sensor data: {n_rows} linhas, {missing_rate*100:.0f}% missing ({pattern})")
        
        timestamps = pd.date_range('2024-01-01', periods=n_rows, freq='h')
        
        # Gerar temperatura com tendência + sazonalidade
        temperature = 20 + np.linspace(0, 5, n_rows) + 5 * np.sin(np.linspace(0, 4*np.pi, n_rows))
        temperature += np.random.normal(0, 1, n_rows)
        
        # Umidade correlacionada negativamente com temperatura
        humidity = 80 - 1.5 * (temperature - 20) + np.random.normal(0, 3, n_rows)
        humidity = np.clip(humidity, 0, 100)
        
        # Pressão relativamente estável
        pressure = 1013 + np.random.normal(0, 2, n_rows)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure
        })
        
        # Introduzir missing values
        numeric_cols = ['temperature', 'humidity', 'pressure']
        n_missing = int(n_rows * missing_rate)
        
        if pattern == "random":
            for col in numeric_cols:
                missing_idx = np.random.choice(n_rows, size=n_missing, replace=False)
                df.loc[missing_idx, col] = np.nan
                
        elif pattern == "temporal":
            # Blocos consecutivos (simula falha de comunicação)
            for col in numeric_cols:
                start_idx = np.random.randint(0, n_rows - n_missing)
                df.loc[start_idx:start_idx+n_missing, col] = np.nan
                
        elif pattern == "column":
            # Uma coluna inteira faltante
            col_to_drop = np.random.choice(numeric_cols, size=1)[0]
            df[col_to_drop] = np.nan
        
        return df
    
    @staticmethod
    def generate_complex_relationships(n_rows: int = 200, missing_rate: float = 0.2):
        """Gera dados com relações complexas entre variáveis"""
        print(f"   Gerando dados com relações complexas")
        
        timestamps = pd.date_range('2024-01-01', periods=n_rows, freq='h')
        
        # Variáveis interdependentes de forma não-linear
        x1 = np.random.randn(n_rows) * 10 + 50
        x2 = 2 * x1 + np.random.randn(n_rows) * 5  # Correlação linear
        x3 = np.sin(x1 / 10) * 20 + np.random.randn(n_rows) * 3  # Relação não-linear
        x4 = (x1 ** 2) / 100 + np.random.randn(n_rows) * 5  # Relação quadrática
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'var1': x1,
            'var2': x2,
            'var3': x3,
            'var4': x4
        })
        
        # Adicionar missing values
        n_missing = int(n_rows * missing_rate)
        for col in ['var1', 'var2', 'var3', 'var4']:
            missing_idx = np.random.choice(n_rows, size=n_missing, replace=False)
            df.loc[missing_idx, col] = np.nan
        
        return df
    
    @staticmethod
    def generate_small_dataset(n_rows: int = 50, missing_rate: float = 0.3):
        """Dataset pequeno (para testar métodos computacionalmente caros)"""
        print(f"   Gerando dataset pequeno: {n_rows} linhas")
        
        timestamps = pd.date_range('2024-01-01', periods=n_rows, freq='h')
        
        data = {
            'timestamp': timestamps,
            'sensor_a': np.random.randn(n_rows) * 5 + 25,
            'sensor_b': np.random.randn(n_rows) * 3 + 15
        }
        
        df = pd.DataFrame(data)
        
        # Adicionar missing
        n_missing = int(n_rows * missing_rate)
        for col in ['sensor_a', 'sensor_b']:
            missing_idx = np.random.choice(n_rows, size=n_missing, replace=False)
            df.loc[missing_idx, col] = np.nan
        
        return df


class ImputatorValidator:
    """Valida decisões do ImputatorNode"""
    
    VALID_METHODS = ["knn", "mice", "gp"]
    
    @staticmethod
    def validate_method_choice(
        dataset_characteristics: Dict,
        chosen_method: str,
        params: Dict
    ) -> Dict:
        """
        Valida se o método escolhido é apropriado para o dataset
        
        Heurísticas:
        - KNN: bom para dados locais, computacionalmente barato
        - MICE: bom para relações complexas, robusto
        - GP: bom para séries temporais pequenas, caro computacionalmente
        """
        score = 0
        max_score = 3
        details = []
        
        n_rows = dataset_characteristics.get('n_rows', 0)
        missing_pattern = dataset_characteristics.get('pattern', 'unknown')
        
        # 1. Método está na lista de válidos?
        if chosen_method in ImputatorValidator.VALID_METHODS:
            score += 1
            details.append(f"✓ Método '{chosen_method}' é válido")
        else:
            details.append(f"✗ Método '{chosen_method}' não é válido")
            return {
                "score": 0,
                "details": details,
                "passed": False,
                "reason": f"Método inválido: {chosen_method}"
            }
        
        # 2. Parâmetros fornecidos?
        if params:
            score += 1
            details.append(f"✓ Parâmetros fornecidos: {params}")
        else:
            details.append("⚠ Nenhum parâmetro fornecido (usando defaults)")
        
        # 3. Escolha apropriada baseada em heurísticas?
        appropriate = False
        
        if chosen_method == "knn":
            # KNN é bom para dados locais e datasets médios/grandes
            if n_rows >= 100 and missing_pattern in ["random", "temporal"]:
                appropriate = True
                details.append("✓ KNN apropriado: dataset médio/grande com padrão local")
            else:
                details.append("⚠ KNN pode não ser ideal para este cenário")
        
        elif chosen_method == "mice":
            # MICE é bom para relações complexas
            if n_rows >= 50:
                appropriate = True
                details.append("✓ MICE apropriado: robusto para relações complexas")
            else:
                details.append("⚠ MICE pode ser excessivo para dataset muito pequeno")
        
        elif chosen_method == "gp":
            # GP é caro, melhor para datasets pequenos
            if n_rows <= 200:
                appropriate = True
                details.append("✓ GP apropriado: dataset pequeno, boa para incerteza")
            else:
                details.append("⚠ GP pode ser muito caro para dataset grande")
        
        if appropriate:
            score += 1
        
        return {
            "score": score / max_score,
            "details": details,
            "passed": score >= 2,  # 66% mínimo
            "chosen_method": chosen_method,
            "params": params
        }
    
    @staticmethod
    def validate_json_output(raw_output: str) -> Dict:
        """Valida se o output é JSON válido"""
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        
        if not json_match:
            return {
                "valid_json": False,
                "reason": "Nenhum JSON encontrado no output"
            }
        
        try:
            data = json.loads(json_match.group(0))
            
            has_method = "method" in data
            has_params = "params" in data
            
            return {
                "valid_json": True,
                "has_method": has_method,
                "has_params": has_params,
                "data": data
            }
        except json.JSONDecodeError as e:
            return {
                "valid_json": False,
                "reason": f"JSON inválido: {e}"
            }


@dataclass
class ImputatorTestResult:
    test_name: str
    test_category: str
    success: bool
    execution_time: float
    dataset_characteristics: Dict
    chosen_method: str
    chosen_params: Dict
    validation_result: Dict
    error_message: str

class ImputatorNodeTester:
    def __init__(self):
        print("\n" + "="*80)
        print("INICIALIZANDO TESTADOR DO IMPUTATOR NODE")
        print("="*80)
        
        load_dotenv()
        
        print("\n[1/3] Inicializando LLM...")
        self.llm = ChatDeepInfra(model="Qwen/Qwen2.5-72B-Instruct")
        print("   ✓ LLM inicializado")
        
        print("\n[2/3] Criando dataset inicial...")
        dummy_df = ImputatorDatasetGenerator.generate_sensor_data_missing(n_rows=10, missing_rate=0.2)
        dummy_path = "temp_dummy_imputator.csv"
        dummy_df.to_csv(dummy_path, index=False)
        print(f"   ✓ Dataset salvo em {dummy_path}")
        
        print("\n[3/3] Inicializando WorkflowExecutor...")
        self.executor = WorkflowExecutor(
            csv_path=dummy_path,
            plot_images_path="./test_plots",
            llm=self.llm
        )
        print("   ✓ Executor inicializado")
        
        self.results: List[ImputatorTestResult] = []
        self.validator = ImputatorValidator()
        
        print("\n✓ Testador pronto!")
    
    def run_all_tests(self):
        print("\n" + "="*80)
        print("EXECUTANDO BATERIA DE TESTES - IMPUTATOR NODE")
        print("="*80)
        
        # CATEGORIA 1: Dados de Sensor (padrão local)
        print("\n[CATEGORIA 1/4] Dados de Sensor com Padrão Local")
        print("-" * 80)
        self._test_sensor_random_missing()
        self._test_sensor_temporal_missing()
        
        # CATEGORIA 2: Relações Complexas
        print("\n[CATEGORIA 2/4] Dados com Relações Complexas")
        print("-" * 80)
        self._test_complex_relationships()
        
        # CATEGORIA 3: Dataset Pequeno (GP recomendado)
        print("\n[CATEGORIA 3/4] Dataset Pequeno")
        print("-" * 80)
        self._test_small_dataset()
        
        # CATEGORIA 4: Diferentes Taxas de Missing
        print("\n[CATEGORIA 4/4] Diferentes Taxas de Missing")
        print("-" * 80)
        self._test_light_missing()
        self._test_heavy_missing()
        
        print("\n" + "="*80)
        print("TESTES CONCLUÍDOS")
        print("="*80)
    
    def _run_single_test(
        self,
        test_name: str,
        test_category: str,
        df: pd.DataFrame,
        context_description: str,
        expected_method: str = None
    ):
        print(f"\n   [{test_category}] {test_name}")
        print(f"   Dataset: {len(df)} linhas, missing: {df.isna().sum().sum()} valores")
        print(f"   Contexto: {context_description[:60]}...")
        
        characteristics = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "missing_count": int(df.isna().sum().sum()),
            "missing_rate": float(df.isna().sum().sum() / (len(df) * len(df.columns))),
            "pattern": test_category.lower()
        }
        
        # Substituir DataFrame
        self.executor.df = df.copy()
        
        start_time = time.time()
        success = False
        error_message = ""
        chosen_method = ""
        chosen_params = {}
        validation_result = {}
        
        try:
            # Criar nó e estado
            node = ImputatorNode(self.executor)
            state = AgentState(msg=context_description, logs=[])
            
            # Executar
            result = node.execute(state)
            execution_time = time.time() - start_time
            
            report = result.get("subagents_report", "")
            
            # Extrair método escolhido do relatório
            # Formato esperado: "Imputator agent decided on method 'knn' with params {...}"
            method_match = re.search(r"method '([^']+)'", report)
            params_match = re.search(r"params ({[^}]+})", report)
            
            if method_match:
                chosen_method = method_match.group(1)
                
                if params_match:
                    try:
                        chosen_params = json.loads(params_match.group(1))
                    except:
                        chosen_params = {}
                
                # Validar escolha
                validation_result = self.validator.validate_method_choice(
                    characteristics,
                    chosen_method,
                    chosen_params
                )
                
                success = validation_result.get("passed", False)
            else:
                error_message = "Não foi possível extrair método do relatório"
                success = False
            
            status = "✓ SUCESSO" if success else "✗ FALHA"
            print(f"   {status} - Tempo: {execution_time:.2f}s - Método: {chosen_method}")
            
            if validation_result:
                print(f"      Score: {validation_result.get('score', 0):.1%}")
            
        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            error_message = str(e)
            print(f"   ✗ ERRO - {error_message}")
        
        # Salvar resultado
        result = ImputatorTestResult(
            test_name=test_name,
            test_category=test_category,
            success=success,
            execution_time=execution_time,
            dataset_characteristics=characteristics,
            chosen_method=chosen_method,
            chosen_params=chosen_params,
            validation_result=validation_result,
            error_message=error_message
        )
        self.results.append(result)
    
    
    def _test_sensor_random_missing(self):
        """Teste: sensor data com missing aleatório"""
        df = ImputatorDatasetGenerator.generate_sensor_data_missing(
            n_rows=200, missing_rate=0.2, pattern="random"
        )
        self._run_single_test(
            test_name="sensor_random_20pct",
            test_category="SensorLocal",
            df=df,
            context_description="The inspection revealed missing data in 'temperature', 'humidity', and 'pressure' columns. These are sensor readings with local patterns and likely have correlations with nearby time points.",
            expected_method="knn"
        )
    
    def _test_sensor_temporal_missing(self):
        """Teste: sensor data com blocos temporais faltantes"""
        df = ImputatorDatasetGenerator.generate_sensor_data_missing(
            n_rows=200, missing_rate=0.15, pattern="temporal"
        )
        self._run_single_test(
            test_name="sensor_temporal_15pct",
            test_category="SensorLocal",
            df=df,
            context_description="Sensor data with temporal blocks of missing values, indicating communication failures. Data shows local patterns in time series.",
            expected_method="knn"
        )
    
    def _test_complex_relationships(self):
        """Teste: relações complexas entre variáveis"""
        df = ImputatorDatasetGenerator.generate_complex_relationships(
            n_rows=200, missing_rate=0.25
        )
        self._run_single_test(
            test_name="complex_relationships",
            test_category="ComplexRelations",
            df=df,
            context_description="Missing data found in 'var1', 'var2', 'var3', and 'var4' columns. These variables are likely interdependent in a complex, non-linear way with multiple relationships.",
            expected_method="mice"
        )
    
    def _test_small_dataset(self):
        """Teste: dataset pequeno (GP recomendado)"""
        df = ImputatorDatasetGenerator.generate_small_dataset(
            n_rows=50, missing_rate=0.3
        )
        self._run_single_test(
            test_name="small_dataset_50_rows",
            test_category="SmallDataset",
            df=df,
            context_description="Small time series dataset (50 rows) with missing values. Uncertainty estimation is important for this safety-critical application.",
            expected_method="gp"
        )
    
    def _test_light_missing(self):
        """Teste: poucos dados faltantes (10%)"""
        df = ImputatorDatasetGenerator.generate_sensor_data_missing(
            n_rows=200, missing_rate=0.1, pattern="random"
        )
        self._run_single_test(
            test_name="light_missing_10pct",
            test_category="MissingRate",
            df=df,
            context_description="Dataset with only 10% missing values in sensor readings. Simple imputation should suffice."
        )
    
    def _test_heavy_missing(self):
        """Teste: muitos dados faltantes (50%)"""
        df = ImputatorDatasetGenerator.generate_sensor_data_missing(
            n_rows=200, missing_rate=0.5, pattern="random"
        )
        self._run_single_test(
            test_name="heavy_missing_50pct",
            test_category="MissingRate",
            df=df,
            context_description="Dataset with 50% missing values across multiple sensor columns. Robust imputation method required."
        )
    
    
    def generate_report(self):
        print("\n" + "="*80)
        print("RELATÓRIO DE RESULTADOS - IMPUTATOR NODE")
        print("="*80)
        
        total = len(self.results)
        success = sum(1 for r in self.results if r.success)
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
        
        # Análise por método escolhido
        methods_chosen = {}
        for r in self.results:
            method = r.chosen_method
            if method:
                methods_chosen[method] = methods_chosen.get(method, 0) + 1
        
        print(f"\n📊 RESUMO GERAL:")
        print(f"   Total de testes: {total}")
        print(f"   Sucessos: {success} ({success_rate:.1f}%)")
        print(f"   Falhas: {total-success} ({100-success_rate:.1f}%)")
        print(f"   Tempo médio: {avg_time:.2f}s")
        
        print(f"\n📊 RESUMO POR CATEGORIA:")
        for cat, stats in categories.items():
            cat_rate = (stats["success"] / stats["total"] * 100)
            print(f"   {cat:20s}: {stats['success']}/{stats['total']} ({cat_rate:.1f}%)")
        
        print(f"\n📊 MÉTODOS ESCOLHIDOS:")
        for method, count in methods_chosen.items():
            print(f"   {method.upper():10s}: {count} vezes")
        
        print(f"\n📋 DETALHES POR TESTE:")
        print("-" * 80)
        for r in self.results:
            status = "✓" if r.success else "✗"
            print(f"{status} [{r.test_category:15s}] {r.test_name:30s} | "
                  f"Método: {r.chosen_method:6s} | "
                  f"Time: {r.execution_time:5.2f}s")
            if not r.success:
                print(f"   └─ {r.error_message}")
        
        return {
            "summary": {
                "total_tests": total,
                "successful": success,
                "failed": total - success,
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
            "methods_chosen": methods_chosen,
            "details": [asdict(r) for r in self.results]
        }
    
    def save_report(self, filename="imputator_node_report.json"):
        report = self.generate_report()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Relatório salvo em: {filename}")
    
    def plot_results(self, filename="imputator_node_plots.png"):
        print(f"\n📊 Gerando gráficos...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Gráfico 1: Taxa de sucesso
        success_count = sum(1 for r in self.results if r.success)
        fail_count = len(self.results) - success_count
        
        axes[0, 0].bar(['Sucesso', 'Falha'], [success_count, fail_count],
                      color=['green', 'red'], alpha=0.7)
        axes[0, 0].set_ylabel('Número de Testes')
        axes[0, 0].set_title('Resultado Geral')
        axes[0, 0].set_ylim([0, len(self.results) + 1])
        
        for i, v in enumerate([success_count, fail_count]):
            axes[0, 0].text(i, v + 0.1, str(v), ha='center', fontweight='bold')
        
        # Gráfico 2: Métodos escolhidos
        methods = {}
        for r in self.results:
            if r.chosen_method:
                methods[r.chosen_method] = methods.get(r.chosen_method, 0) + 1
        
        if methods:
            axes[0, 1].bar(methods.keys(), methods.values(), color='steelblue', alpha=0.7)
            axes[0, 1].set_ylabel('Frequência')
            axes[0, 1].set_title('Métodos de Imputação Escolhidos')
            axes[0, 1].tick_params(axis='x', rotation=0)
        
        # Gráfico 3: Tempo por categoria
        cat_times = {}
        for r in self.results:
            cat = r.test_category
            if cat not in cat_times:
                cat_times[cat] = []
            cat_times[cat].append(r.execution_time)
        
        cat_names = list(cat_times.keys())
        cat_avg_times = [np.mean(cat_times[c]) for c in cat_names]
        
        axes[1, 0].barh(cat_names, cat_avg_times, color='coral', alpha=0.7)
        axes[1, 0].set_xlabel('Tempo Médio (s)')
        axes[1, 0].set_title('Tempo por Categoria')
        axes[1, 0].invert_yaxis()
        
        # Gráfico 4: Taxa de sucesso por categoria
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
        
        axes[1, 1].bar(cat_names, cat_rates, color='mediumseagreen', alpha=0.7)
        axes[1, 1].set_ylabel('Taxa de Sucesso (%)')
        axes[1, 1].set_title('Sucesso por Categoria')
        axes[1, 1].set_ylim([0, 105])
        axes[1, 1].axhline(y=80, color='r', linestyle='--', label='Meta: 80%')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Gráficos salvos em: {filename}")
        plt.close()


def main():
    print("\n" + "="*80)
    print(" SISTEMA DE TESTES DO IMPUTATOR NODE ".center(80, "="))
    print("="*80)
    
    try:
        tester = ImputatorNodeTester()
        tester.run_all_tests()
        tester.generate_report()
        tester.save_report()
        tester.plot_results()
        
        print("\n" + "="*80)
        print(" TESTES CONCLUÍDOS COM SUCESSO ".center(80, "="))
        print("="*80)
        print("\n📁 Arquivos gerados:")
        print("   - imputator_node_report.json")
        print("   - imputator_node_plots.png")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()