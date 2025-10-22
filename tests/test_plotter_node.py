"""
Sistema de Avaliação do PlotterNode
Arquivo: test_plotter_node.py

OBJETIVO: Testar se o PlotterNode cria visualizações corretamente

COMO USAR:
python test_plotter_node.py
"""

import pandas as pd
import numpy as np
import time
import json
import os
import sys
import shutil
from typing import Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Adiciona o diretório do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agentai.workflow import WorkflowExecutor
from agentai.nodes import PlotterNode
from agentai.modules.common import AgentState
from langchain_community.chat_models import ChatDeepInfra
from dotenv import load_dotenv


class PlotterDatasetGenerator:
    """Gera datasets para testar visualizações"""
    
    @staticmethod
    def generate_timeseries(n_rows=200):
        """Gera série temporal simples"""
        print(f"   Gerando série temporal: {n_rows} pontos")
        
        timestamps = pd.date_range('2024-01-01', periods=n_rows, freq='h')
        
        temperature = 20 + 5*np.sin(np.linspace(0, 4*np.pi, n_rows)) + np.random.randn(n_rows)
        humidity = 60 - 1.5*(temperature-20) + np.random.randn(n_rows)*3
        pressure = 1013 + np.random.randn(n_rows)*2
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure
        })
    
    @staticmethod
    def generate_multivariate(n_rows=200, n_sensors=5):
        """Gera dados multivariados para correlações"""
        print(f"   Gerando dados multivariados: {n_sensors} sensores")
        
        timestamps = pd.date_range('2024-01-01', periods=n_rows, freq='h')
        
        data = {'timestamp': timestamps}
        
        # Sensores correlacionados
        base = np.random.randn(n_rows) * 10 + 25
        for i in range(n_sensors):
            data[f'sensor_{i}'] = base + np.random.randn(n_rows) * (i+1)
        
        return pd.DataFrame(data)


class PlotterValidator:
    """Valida se plots foram criados"""
    
    @staticmethod
    def validate_files_created(
        images_path: str,
        is_before_dp: bool,
        expected_count: int = 1
    ) -> Dict:
        """Valida se arquivos PNG foram criados no diretório correto"""
        
        # Determinar subdiretório
        subdir = "before_dp" if is_before_dp else "after_dp"
        full_path = os.path.join(images_path, subdir)
        
        # Verificar se diretório existe
        if not os.path.exists(full_path):
            return {
                "passed": False,
                "score": 0.0,
                "reason": f"✗ Diretório não existe: {full_path}",
                "files_found": []
            }
        
        # Listar arquivos PNG
        all_files = os.listdir(full_path)
        png_files = [f for f in all_files if f.endswith('.png')]
        
        if len(png_files) >= expected_count:
            return {
                "passed": True,
                "score": 1.0,
                "reason": f"✓ {len(png_files)} arquivo(s) PNG criado(s)",
                "files_found": png_files
            }
        else:
            return {
                "passed": False,
                "score": len(png_files) / expected_count if expected_count > 0 else 0,
                "reason": f"✗ Esperava {expected_count} arquivo(s), encontrou {len(png_files)}",
                "files_found": png_files
            }
    
    @staticmethod
    def validate_no_errors_in_output(output: str) -> Dict:
        """Valida se não há erros no output do agente"""
        
        error_keywords = ["error", "failed", "exception", "traceback"]
        
        output_lower = output.lower()
        found_errors = [kw for kw in error_keywords if kw in output_lower]
        
        if found_errors:
            return {
                "passed": False,
                "score": 0.0,
                "reason": f"✗ Erros encontrados: {found_errors}"
            }
        else:
            return {
                "passed": True,
                "score": 1.0,
                "reason": "✓ Nenhum erro no output"
            }


@dataclass
class PlotterTestResult:
    test_name: str
    test_category: str
    success: bool
    execution_time: float
    instruction: str
    files_created: List[str]
    validation_results: Dict
    error_message: str

class PlotterNodeTester:
    def __init__(self):
        print("\n" + "="*80)
        print("INICIALIZANDO TESTADOR DO PLOTTER NODE")
        print("="*80)
        
        load_dotenv()
        
        print("\n[1/4] Inicializando LLM...")
        self.llm = ChatDeepInfra(model="Qwen/Qwen2.5-72B-Instruct")
        print("   ✓ LLM inicializado")
        
        print("\n[2/4] Criando dataset inicial...")
        dummy_df = PlotterDatasetGenerator.generate_timeseries(n_rows=50)
        dummy_path = "temp_dummy_plotter.csv"
        dummy_df.to_csv(dummy_path, index=False)
        print(f"   ✓ Dataset salvo em {dummy_path}")
        
        print("\n[3/4] Configurando diretório de imagens...")
        self.images_path = "./test_plots_plotter"
        # Limpar diretório anterior se existir
        if os.path.exists(self.images_path):
            shutil.rmtree(self.images_path)
        os.makedirs(self.images_path, exist_ok=True)
        print(f"   ✓ Diretório criado: {self.images_path}")
        
        print("\n[4/4] Inicializando WorkflowExecutor...")
        self.executor = WorkflowExecutor(
            csv_path=dummy_path,
            plot_images_path=self.images_path,
            llm=self.llm
        )
        print("   ✓ Executor inicializado")
        
        self.results: List[PlotterTestResult] = []
        self.validator = PlotterValidator()
        
        print("\n✓ Testador pronto!")
    
    def run_all_tests(self):
        print("\n" + "="*80)
        print("EXECUTANDO BATERIA DE TESTES - PLOTTER NODE")
        print("="*80)
        
        # CATEGORIA 1: Plots Básicos
        print("\n[CATEGORIA 1/4] Plots Básicos")
        print("-" * 80)
        self._test_time_series_plot()
        self._test_scatter_plot()
        self._test_histogram()
        
        # CATEGORIA 2: Plots de Correlação
        print("\n[CATEGORIA 2/4] Análise de Correlação")
        print("-" * 80)
        self._test_heatmap()
        
        # CATEGORIA 3: Instruções Específicas
        print("\n[CATEGORIA 3/4] Instruções Específicas do Usuário")
        print("-" * 80)
        self._test_specific_column_plot()
        self._test_multiple_plots()
        
        # CATEGORIA 4: is_before_dp flag
        print("\n[CATEGORIA 4/4] Teste de Diretórios (before_dp vs after_dp)")
        print("-" * 80)
        self._test_before_dp_directory()
        self._test_after_dp_directory()
        
        print("\n" + "="*80)
        print("TESTES CONCLUÍDOS")
        print("="*80)
    
    def _run_single_test(
        self,
        test_name: str,
        test_category: str,
        df: pd.DataFrame,
        instruction: str,
        is_before_dp: bool = True
    ):
        print(f"\n   [{test_category}] {test_name}")
        print(f"   Dataset: {len(df)} linhas, {len(df.columns)} colunas")
        print(f"   Instrução: {instruction[:60]}...")
        
        # Substituir DataFrame
        self.executor.df = df.copy()
        
        # Limpar diretório de plots antes do teste
        subdir = "before_dp" if is_before_dp else "after_dp"
        full_path = os.path.join(self.images_path, subdir)
        if os.path.exists(full_path):
            shutil.rmtree(full_path)
        os.makedirs(full_path, exist_ok=True)
        
        start_time = time.time()
        success = False
        error_message = ""
        files_created = []
        validation_results = {}
        
        try:
            # Criar nó e executar
            node = PlotterNode(self.executor)
            state = AgentState(
                msg=instruction,
                logs=[],
                is_before_dp=is_before_dp
            )
            
            result = node.execute(state)
            execution_time = time.time() - start_time
            
            output = result.get("subagents_report", "")
            
            # Validar
            val_files = self.validator.validate_files_created(
                self.images_path,
                is_before_dp,
                expected_count=1
            )
            
            val_errors = self.validator.validate_no_errors_in_output(output)
            
            validation_results = {
                "files_created": val_files,
                "no_errors": val_errors
            }
            
            files_created = val_files.get("files_found", [])
            
            # Sucesso se ambos passarem
            success = val_files["passed"] and val_errors["passed"]
            
            status = "✓ SUCESSO" if success else "✗ FALHA"
            print(f"   {status} - Tempo: {execution_time:.2f}s - Arquivos: {len(files_created)}")
            print(f"      {val_files['reason']}")
            print(f"      {val_errors['reason']}")
            
        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            error_message = str(e)
            print(f"   ✗ ERRO - {error_message}")
        
        # Salvar resultado
        result = PlotterTestResult(
            test_name=test_name,
            test_category=test_category,
            success=success,
            execution_time=execution_time,
            instruction=instruction,
            files_created=files_created,
            validation_results=validation_results,
            error_message=error_message
        )
        self.results.append(result)
    
    
    def _test_time_series_plot(self):
        """Teste: plot de série temporal"""
        df = PlotterDatasetGenerator.generate_timeseries(n_rows=200)
        self._run_single_test(
            test_name="time_series_basic",
            test_category="BasicPlots",
            df=df,
            instruction="Create a time series plot showing temperature over time.",
            is_before_dp=True
        )
    
    def _test_scatter_plot(self):
        """Teste: scatter plot"""
        df = PlotterDatasetGenerator.generate_timeseries(n_rows=200)
        self._run_single_test(
            test_name="scatter_temp_humidity",
            test_category="BasicPlots",
            df=df,
            instruction="Create a scatter plot showing the relationship between temperature and humidity.",
            is_before_dp=True
        )
    
    def _test_histogram(self):
        """Teste: histograma"""
        df = PlotterDatasetGenerator.generate_timeseries(n_rows=200)
        self._run_single_test(
            test_name="histogram_distribution",
            test_category="BasicPlots",
            df=df,
            instruction="Create histograms to show the distribution of all numeric variables.",
            is_before_dp=True
        )
    
    def _test_heatmap(self):
        """Teste: heatmap de correlação"""
        df = PlotterDatasetGenerator.generate_multivariate(n_rows=200, n_sensors=5)
        self._run_single_test(
            test_name="correlation_heatmap",
            test_category="Correlation",
            df=df,
            instruction="Create a correlation heatmap for all sensors.",
            is_before_dp=True
        )
    
    def _test_specific_column_plot(self):
        """Teste: plot de coluna específica"""
        df = PlotterDatasetGenerator.generate_timeseries(n_rows=200)
        self._run_single_test(
            test_name="specific_column_temp",
            test_category="SpecificInstructions",
            df=df,
            instruction="Plot only the temperature column over time.",
            is_before_dp=True
        )
    
    def _test_multiple_plots(self):
        """Teste: múltiplos plots"""
        df = PlotterDatasetGenerator.generate_timeseries(n_rows=200)
        self._run_single_test(
            test_name="multiple_plots_request",
            test_category="SpecificInstructions",
            df=df,
            instruction="Create multiple plots: time series for temperature, scatter for temp vs humidity, and a histogram.",
            is_before_dp=True
        )
    
    def _test_before_dp_directory(self):
        """Teste: plots vão para before_dp"""
        df = PlotterDatasetGenerator.generate_timeseries(n_rows=100)
        self._run_single_test(
            test_name="before_dp_directory",
            test_category="DirectoryHandling",
            df=df,
            instruction="Create a simple time series plot.",
            is_before_dp=True
        )
    
    def _test_after_dp_directory(self):
        """Teste: plots vão para after_dp"""
        df = PlotterDatasetGenerator.generate_timeseries(n_rows=100)
        self._run_single_test(
            test_name="after_dp_directory",
            test_category="DirectoryHandling",
            df=df,
            instruction="Create a simple time series plot.",
            is_before_dp=False
        )
    
    
    def generate_report(self):
        print("\n" + "="*80)
        print("RELATÓRIO DE RESULTADOS - PLOTTER NODE")
        print("="*80)
        
        total = len(self.results)
        success = sum(1 for r in self.results if r.success)
        success_rate = (success / total * 100) if total > 0 else 0
        avg_time = np.mean([r.execution_time for r in self.results])
        total_files = sum(len(r.files_created) for r in self.results)
        
        # Por categoria
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
        print(f"   Falhas: {total-success} ({100-success_rate:.1f}%)")
        print(f"   Tempo médio: {avg_time:.2f}s")
        print(f"   Total de arquivos criados: {total_files}")
        
        print(f"\n📊 RESUMO POR CATEGORIA:")
        for cat, stats in categories.items():
            cat_rate = (stats["success"] / stats["total"] * 100)
            print(f"   {cat:20s}: {stats['success']}/{stats['total']} ({cat_rate:.1f}%)")
        
        print(f"\n📋 DETALHES POR TESTE:")
        print("-" * 80)
        for r in self.results:
            status = "✓" if r.success else "✗"
            print(f"{status} [{r.test_category:20s}] {r.test_name:30s} | "
                  f"Files: {len(r.files_created)} | "
                  f"Time: {r.execution_time:5.2f}s")
            if r.files_created:
                print(f"   └─ Criados: {', '.join(r.files_created[:3])}")
        
        return {
            "summary": {
                "total_tests": total,
                "successful": success,
                "failed": total - success,
                "success_rate": f"{success_rate:.1f}%",
                "average_time": f"{avg_time:.2f}s",
                "total_files_created": total_files
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
    
    def save_report(self, filename="plotter_node_report.json"):
        report = self.generate_report()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Relatório salvo em: {filename}")
    
    def plot_results(self, filename="plotter_node_plots.png"):
        print(f"\n📊 Gerando gráficos...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Gráfico 1: Sucesso vs Falha
        success_count = sum(1 for r in self.results if r.success)
        fail_count = len(self.results) - success_count
        
        axes[0, 0].bar(['Sucesso', 'Falha'], [success_count, fail_count],
                      color=['green', 'red'], alpha=0.7)
        axes[0, 0].set_ylabel('Número de Testes')
        axes[0, 0].set_title('Resultado Geral')
        
        for i, v in enumerate([success_count, fail_count]):
            axes[0, 0].text(i, v + 0.1, str(v), ha='center', fontweight='bold')
        
        # Gráfico 2: Arquivos criados por teste
        test_names = [r.test_name[:15] for r in self.results]
        file_counts = [len(r.files_created) for r in self.results]
        colors = ['green' if r.success else 'red' for r in self.results]
        
        axes[0, 1].barh(test_names, file_counts, color=colors, alpha=0.7)
        axes[0, 1].set_xlabel('Número de Arquivos')
        axes[0, 1].set_title('Arquivos Criados por Teste')
        axes[0, 1].invert_yaxis()
        
        # Gráfico 3: Tempo de execução
        times = [r.execution_time for r in self.results]
        
        axes[1, 0].barh(test_names, times, color=colors, alpha=0.7)
        axes[1, 0].set_xlabel('Tempo (s)')
        axes[1, 0].set_title('Tempo de Execução')
        axes[1, 0].invert_yaxis()
        
        # Gráfico 4: Taxa por categoria
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
    print(" SISTEMA DE TESTES DO PLOTTER NODE ".center(80, "="))
    print("="*80)
    
    try:
        tester = PlotterNodeTester()
        tester.run_all_tests()
        tester.generate_report()
        tester.save_report()
        tester.plot_results()
        
        print("\n" + "="*80)
        print(" TESTES CONCLUÍDOS COM SUCESSO ".center(80, "="))
        print("="*80)
        print("\n📁 Arquivos gerados:")
        print("   - plotter_node_report.json")
        print("   - plotter_node_plots.png")
        print(f"   - Plots criados em: ./test_plots_plotter/\n")
        
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()