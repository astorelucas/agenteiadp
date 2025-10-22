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

# Adiciona o diretório do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from agentai.workflow import WorkflowExecutor
from agentai.nodes import PandasNode
from agentai.modules.common import AgentState
from langchain_community.chat_models import ChatDeepInfra
from dotenv import load_dotenv


class DatasetGenerator:
    """Gera datasets sintéticos para teste"""
    
    @staticmethod
    def generate_simple_dataset(
        n_rows: int = 100,
        n_features: int = 3,
        missing_rate: float = 0.0,
        has_outliers: bool = False,
        has_infinity: bool = False
    ) -> pd.DataFrame:
        """
        Gera um dataset simples para testes
        """
        print(f"   Gerando dataset: {n_rows} linhas, {n_features} features, "
              f"missing={missing_rate:.1%}, outliers={has_outliers}, inf={has_infinity}")
        
        # Criar timestamps
        start_date = datetime(2024, 1, 1)
        timestamps = [start_date + timedelta(hours=i) for i in range(n_rows)]
        
        # Criar features numéricas
        data = {'timestamp': timestamps}
        
        for i in range(n_features):
            # Série temporal simples: tendência + ruído
            trend = np.linspace(20, 30, n_rows)
            noise = np.random.normal(0, 2, n_rows)
            data[f'feature_{i}'] = trend + noise
        
        df = pd.DataFrame(data)
        
        # Adicionar outliers se necessário
        if has_outliers:
            n_outliers = max(1, int(n_rows * 0.05))  # 5% de outliers
            outlier_indices = np.random.choice(n_rows, size=n_outliers, replace=False)
            for col in df.select_dtypes(include='number').columns:
                df.loc[outlier_indices, col] = df[col].mean() + 10 * df[col].std()
        
        # Adicionar infinitos se necessário
        if has_infinity:
            n_inf = max(1, int(n_rows * 0.02))  # 2% de infinitos
            inf_indices = np.random.choice(n_rows, size=n_inf, replace=False)
            for col in df.select_dtypes(include='number').columns:
                df.loc[inf_indices[:n_inf//2], col] = np.inf
                if n_inf > 1:
                    df.loc[inf_indices[n_inf//2:], col] = -np.inf
        
        # Adicionar dados faltantes se necessário
        if missing_rate > 0:
            numeric_cols = df.select_dtypes(include='number').columns
            n_missing = int(n_rows * missing_rate)
            
            for col in numeric_cols:
                missing_idx = np.random.choice(n_rows, size=n_missing, replace=False)
                df.loc[missing_idx, col] = np.nan
        
        return df


@dataclass
class TestResult:
    """Armazena resultado de um teste"""
    test_name: str
    success: bool
    execution_time: float
    dataset_size: int
    missing_rate: float
    error_message: str
    agent_output: str

class SimpleInspectTester:
    """Classe simplificada para testar o nó Inspect"""
    
    def __init__(self):
        """Inicializa o testador"""
        print("\n" + "="*80)
        print("INICIALIZANDO TESTADOR DO NÓ INSPECT")
        print("="*80)
        
        # Carregar variáveis de ambiente
        load_dotenv()
        
        # Inicializar LLM
        print("\n[1/3] Inicializando LLM...")
        self.llm = ChatDeepInfra(model="Qwen/Qwen2.5-72B-Instruct")
        print("   ✓ LLM inicializado")
        
        # Criar um dataset dummy inicial para inicializar o executor
        print("\n[2/3] Criando dataset inicial...")
        dummy_df = DatasetGenerator.generate_simple_dataset(n_rows=10, n_features=2)
        dummy_path = "temp_dummy_dataset.csv"
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
        
        self.results: List[TestResult] = []
        
        print("\n✓ Testador pronto para executar testes!")
    
    def run_all_tests(self):
        """Executa todos os testes"""
        print("\n" + "="*80)
        print("EXECUTANDO BATERIA DE TESTES")
        print("="*80)
        
        # Teste 1: Dataset limpo (sem problemas)
        print("\n[TESTE 1/6] Dataset limpo (baseline)")
        print("-" * 40)
        self._test_clean_dataset()
        
        # Teste 2: Dataset com poucos dados faltantes
        print("\n[TESTE 2/6] Dataset com 10% de dados faltantes")
        print("-" * 40)
        self._test_light_missing()
        
        # Teste 3: Dataset com muitos dados faltantes
        print("\n[TESTE 3/6] Dataset com 50% de dados faltantes")
        print("-" * 40)
        self._test_heavy_missing()
        
        # Teste 4: Dataset com outliers
        print("\n[TESTE 4/6] Dataset com outliers")
        print("-" * 40)
        self._test_outliers()
        
        # Teste 5: Dataset com valores infinitos
        print("\n[TESTE 5/6] Dataset com valores infinitos")
        print("-" * 40)
        self._test_infinity()
        
        # Teste 6: Dataset grande (teste de escalabilidade)
        print("\n[TESTE 6/6] Dataset grande (1000 linhas)")
        print("-" * 40)
        self._test_large_dataset()
        
        print("\n" + "="*80)
        print("TESTES CONCLUÍDOS")
        print("="*80)
    
    def _run_single_test(self, test_name: str, df: pd.DataFrame, instruction: str):
        """Executa um único teste"""
        print(f"\n   Executando: {test_name}")
        print(f"   Dataset: {len(df)} linhas, {len(df.columns)} colunas")
        print(f"   Instrução: {instruction[:60]}...")
        
        # Substituir DataFrame no executor
        self.executor.df = df.copy()
        
        # Calcular características
        missing_rate = df.isna().sum().sum() / (len(df) * len(df.columns))
        
        # Executar teste
        start_time = time.time()
        success = False
        error_message = ""
        agent_output = ""
        
        try:
            # Criar nó e estado
            node = PandasNode(self.executor)
            state = AgentState(msg=instruction, logs=[])
            
            # Executar
            result = node.execute(state)
            execution_time = time.time() - start_time
            
            # Extrair output
            agent_output = result.get("subagents_report", "")
            
            # Verificar sucesso
            if "error" in agent_output.lower() and "no error" not in agent_output.lower():
                success = False
                error_message = "Agent reported an error"
            else:
                success = True
                error_message = ""
            
            status = "✓ SUCESSO" if success else "✗ FALHA"
            print(f"   {status} - Tempo: {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            error_message = str(e)
            agent_output = ""
            print(f"   ✗ ERRO - {error_message}")
        
        # Salvar resultado
        result = TestResult(
            test_name=test_name,
            success=success,
            execution_time=execution_time,
            dataset_size=len(df),
            missing_rate=missing_rate,
            error_message=error_message,
            agent_output=agent_output[:500]  # Limitar tamanho
        )
        self.results.append(result)
    
    def _test_clean_dataset(self):
        """Teste com dataset limpo"""
        df = DatasetGenerator.generate_simple_dataset(
            n_rows=100, 
            n_features=3, 
            missing_rate=0.0
        )
        self._run_single_test(
            "clean_dataset",
            df,
            "Provide a summary of the dataset including shape, data types, and basic statistics."
        )
    
    def _test_light_missing(self):
        """Teste com poucos dados faltantes"""
        df = DatasetGenerator.generate_simple_dataset(
            n_rows=100, 
            n_features=3, 
            missing_rate=0.1
        )
        self._run_single_test(
            "light_missing_10pct",
            df,
            "Analyze the dataset and identify any missing values. Report the count for each column."
        )
    
    def _test_heavy_missing(self):
        """Teste com muitos dados faltantes"""
        df = DatasetGenerator.generate_simple_dataset(
            n_rows=100, 
            n_features=3, 
            missing_rate=0.5
        )
        self._run_single_test(
            "heavy_missing_50pct",
            df,
            "This dataset has many missing values. Analyze the missing data pattern and suggest appropriate imputation methods."
        )
    
    def _test_outliers(self):
        """Teste com outliers"""
        df = DatasetGenerator.generate_simple_dataset(
            n_rows=100, 
            n_features=3, 
            missing_rate=0.1,
            has_outliers=True
        )
        self._run_single_test(
            "dataset_with_outliers",
            df,
            "Analyze this dataset for outliers and anomalies. Report any suspicious values."
        )
    
    def _test_infinity(self):
        """Teste com valores infinitos"""
        df = DatasetGenerator.generate_simple_dataset(
            n_rows=100, 
            n_features=3, 
            missing_rate=0.1,
            has_infinity=True
        )
        self._run_single_test(
            "dataset_with_infinity",
            df,
            "Check for infinite values and other data quality issues in this dataset."
        )
    
    def _test_large_dataset(self):
        """Teste de escalabilidade"""
        df = DatasetGenerator.generate_simple_dataset(
            n_rows=1000, 
            n_features=5, 
            missing_rate=0.2
        )
        self._run_single_test(
            "large_dataset_1000_rows",
            df,
            "Provide a comprehensive analysis of this dataset."
        )
    
    def generate_report(self):
        """Gera relatório dos resultados"""
        print("\n" + "="*80)
        print("RELATÓRIO DE RESULTADOS")
        print("="*80)
        
        total = len(self.results)
        success = sum(1 for r in self.results if r.success)
        fail = total - success
        
        success_rate = (success / total * 100) if total > 0 else 0
        avg_time = np.mean([r.execution_time for r in self.results])
        
        print(f"\n📊 RESUMO GERAL:")
        print(f"   Total de testes: {total}")
        print(f"   Sucessos: {success} ({success_rate:.1f}%)")
        print(f"   Falhas: {fail} ({100-success_rate:.1f}%)")
        print(f"   Tempo médio: {avg_time:.2f}s")
        
        print(f"\n📋 DETALHES POR TESTE:")
        print("-" * 80)
        for r in self.results:
            status = "✓" if r.success else "✗"
            print(f"{status} {r.test_name:30s} | "
                  f"Size: {r.dataset_size:4d} | "
                  f"Missing: {r.missing_rate:5.1%} | "
                  f"Time: {r.execution_time:5.2f}s")
            if not r.success:
                print(f"   └─ Erro: {r.error_message[:70]}")
        
        return {
            "summary": {
                "total_tests": total,
                "successful": success,
                "failed": fail,
                "success_rate": f"{success_rate:.1f}%",
                "average_time": f"{avg_time:.2f}s"
            },
            "details": [asdict(r) for r in self.results]
        }
    
    def save_report(self, filename="inspect_test_report.json"):
        """Salva relatório em arquivo JSON"""
        report = self.generate_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Relatório salvo em: {filename}")
    
    def plot_results(self, filename="inspect_test_plots.png"):
        """Gera gráficos dos resultados"""
        print(f"\n📊 Gerando gráficos...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico 1: Taxa de sucesso
        success_count = sum(1 for r in self.results if r.success)
        fail_count = len(self.results) - success_count
        
        axes[0].bar(['Sucesso', 'Falha'], [success_count, fail_count], 
                    color=['green', 'red'], alpha=0.7)
        axes[0].set_ylabel('Número de Testes')
        axes[0].set_title('Resultado dos Testes')
        axes[0].set_ylim([0, len(self.results) + 1])
        
        # Adicionar valores nas barras
        for i, v in enumerate([success_count, fail_count]):
            axes[0].text(i, v + 0.1, str(v), ha='center', fontweight='bold')
        
        # Gráfico 2: Tempo de execução
        test_names = [r.test_name.replace('_', '\n') for r in self.results]
        times = [r.execution_time for r in self.results]
        colors = ['green' if r.success else 'red' for r in self.results]
        
        axes[1].barh(test_names, times, color=colors, alpha=0.7)
        axes[1].set_xlabel('Tempo de Execução (s)')
        axes[1].set_title('Tempo por Teste')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Gráficos salvos em: {filename}")
        plt.close()


def main():
    """Função principal para executar os testes"""
    
    print("\n" + "="*80)
    print(" SISTEMA DE TESTES DO NÓ INSPECT ".center(80, "="))
    print("="*80)
    
    try:
        # Inicializar testador
        tester = SimpleInspectTester()
        
        # Executar testes
        tester.run_all_tests()
        
        # Gerar relatório
        tester.generate_report()
        
        # Salvar resultados
        tester.save_report("inspect_test_report.json")
        tester.plot_results("inspect_test_plots.png")
        
        print("\n" + "="*80)
        print(" TESTES CONCLUÍDOS COM SUCESSO ".center(80, "="))
        print("="*80)
        print("\n📁 Arquivos gerados:")
        print("   - inspect_test_report.json (relatório detalhado)")
        print("   - inspect_test_plots.png (gráficos)")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()