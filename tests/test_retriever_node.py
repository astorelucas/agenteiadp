"""
Sistema de Avaliação do RetrieverNode
Arquivo: test_retriever_node.py

OBJETIVO: Testar se o RetrieverNode busca informações relevantes no RAG

COMO USAR:
python test_retriever_node.py
"""

import time
import json
import os
import sys
from typing import Dict, List
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import numpy as np

# Adiciona o diretório do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agentai.nodes import RetrieverNode
from agentai.modules.common import AgentState
from dotenv import load_dotenv


class RetrieverValidator:
    """Valida outputs do RetrieverNode"""
    
    @staticmethod
    def validate_output_exists(output: str) -> Dict:
        """Valida se retornou algum resultado"""
        if not output or len(output) < 20:
            return {
                "passed": False,
                "score": 0.0,
                "reason": "✗ Output vazio ou muito curto"
            }
        return {
            "passed": True,
            "score": 1.0,
            "reason": f"✓ Retornou {len(output)} caracteres"
        }
    
    @staticmethod
    def validate_relevance(query: str, output: str) -> Dict:
        """Valida relevância básica (keywords)"""
        if not query or not output:
            return {
                "passed": False,
                "score": 0.0,
                "reason": "Query ou output vazio"
            }
        
        # Extrair keywords da query (palavras com 4+ chars)
        query_words = set(
            word.lower() for word in query.split() 
            if len(word) >= 4
        )
        
        output_lower = output.lower()
        
        # Contar quantas keywords aparecem no output
        matches = sum(1 for word in query_words if word in output_lower)
        relevance = matches / len(query_words) if query_words else 0
        
        if relevance >= 0.3:  # 30% de overlap
            return {
                "passed": True,
                "score": relevance,
                "reason": f"✓ Relevância: {relevance:.1%} ({matches}/{len(query_words)} keywords)"
            }
        else:
            return {
                "passed": False,
                "score": relevance,
                "reason": f"✗ Baixa relevância: {relevance:.1%}"
            }


@dataclass
class RetrieverTestResult:
    test_name: str
    test_category: str
    success: bool
    execution_time: float
    query: str
    output_length: int
    validation_results: Dict
    error_message: str

class RetrieverNodeTester:
    def __init__(self):
        print("\n" + "="*80)
        print("INICIALIZANDO TESTADOR DO RETRIEVER NODE")
        print("="*80)
        
        load_dotenv()
        print("\n✓ Testador pronto!")
        
        self.results: List[RetrieverTestResult] = []
        self.validator = RetrieverValidator()
    
    def run_all_tests(self):
        print("\n" + "="*80)
        print("EXECUTANDO BATERIA DE TESTES - RETRIEVER NODE")
        print("="*80)
        
        # CATEGORIA 1: Queries sobre Missing Data
        print("\n[CATEGORIA 1/4] Queries sobre Dados Faltantes")
        print("-" * 80)
        self._test_missing_data_query()
        self._test_imputation_methods_query()
        
        # CATEGORIA 2: Queries sobre Feature Engineering
        print("\n[CATEGORIA 2/4] Queries sobre Feature Engineering")
        print("-" * 80)
        self._test_feature_engineering_query()
        self._test_rolling_average_query()
        
        # CATEGORIA 3: Queries sobre Erros
        print("\n[CATEGORIA 3/4] Queries sobre Erros")
        print("-" * 80)
        self._test_error_resolution_query()
        self._test_numpy_import_error()
        
        # CATEGORIA 4: Queries Inválidas
        print("\n[CATEGORIA 4/4] Queries Inválidas/Edge Cases")
        print("-" * 80)
        self._test_empty_query()
        self._test_vague_query()
        
        print("\n" + "="*80)
        print("TESTES CONCLUÍDOS")
        print("="*80)
    
    def _run_single_test(
        self,
        test_name: str,
        test_category: str,
        query: str
    ):
        print(f"\n   [{test_category}] {test_name}")
        print(f"   Query: {query[:60]}...")
        
        start_time = time.time()
        success = False
        error_message = ""
        output = ""
        validation_results = {}
        
        try:
            # Criar nó e executar
            node = RetrieverNode()
            state = AgentState(msg=query, logs=[])
            
            result = node.execute(state)
            execution_time = time.time() - start_time
            
            # Extrair output
            output = result.get("subagents_report", "")
            
            # Validar
            val_exists = self.validator.validate_output_exists(output)
            val_relevance = self.validator.validate_relevance(query, output)
            
            validation_results = {
                "output_exists": val_exists,
                "relevance": val_relevance
            }
            
            # Sucesso se ambos passarem
            success = val_exists["passed"] and val_relevance["passed"]
            
            status = "✓ SUCESSO" if success else "✗ FALHA"
            print(f"   {status} - Tempo: {execution_time:.2f}s - Output: {len(output)} chars")
            print(f"      {val_exists['reason']}")
            print(f"      {val_relevance['reason']}")
            
        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            error_message = str(e)
            print(f"   ✗ ERRO - {error_message}")
        
        # Salvar resultado
        result = RetrieverTestResult(
            test_name=test_name,
            test_category=test_category,
            success=success,
            execution_time=execution_time,
            query=query,
            output_length=len(output),
            validation_results=validation_results,
            error_message=error_message
        )
        self.results.append(result)
    

    
    def _test_missing_data_query(self):
        self._run_single_test(
            "missing_data_general",
            "MissingData",
            "How to handle missing data in time series IoT datasets?"
        )
    
    def _test_imputation_methods_query(self):
        self._run_single_test(
            "imputation_methods",
            "MissingData",
            "What are the best imputation methods for sensor data with local patterns?"
        )
    
    def _test_feature_engineering_query(self):
        self._run_single_test(
            "feature_engineering_iot",
            "FeatureEngineering",
            "Best practices for feature engineering in IoT time series data"
        )
    
    def _test_rolling_average_query(self):
        self._run_single_test(
            "rolling_average_how_to",
            "FeatureEngineering",
            "How to create rolling average features for temperature sensor data?"
        )
    
    def _test_error_resolution_query(self):
        self._run_single_test(
            "error_resolution",
            "ErrorHandling",
            "Agent failed with NameError - how to resolve import errors in pandas code?"
        )
    
    def _test_numpy_import_error(self):
        self._run_single_test(
            "numpy_import_error",
            "ErrorHandling",
            "NameError: name 'np' is not defined - how to fix?"
        )
    
    def _test_empty_query(self):
        self._run_single_test(
            "empty_query",
            "InvalidQueries",
            ""
        )
    
    def _test_vague_query(self):
        self._run_single_test(
            "vague_query",
            "InvalidQueries",
            "help"
        )
    
    
    def generate_report(self):
        print("\n" + "="*80)
        print("RELATÓRIO DE RESULTADOS - RETRIEVER NODE")
        print("="*80)
        
        total = len(self.results)
        success = sum(1 for r in self.results if r.success)
        success_rate = (success / total * 100) if total > 0 else 0
        avg_time = np.mean([r.execution_time for r in self.results])
        avg_output_length = np.mean([r.output_length for r in self.results])
        
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
        print(f"   Tamanho médio output: {avg_output_length:.0f} chars")
        
        print(f"\n📊 RESUMO POR CATEGORIA:")
        for cat, stats in categories.items():
            cat_rate = (stats["success"] / stats["total"] * 100)
            print(f"   {cat:20s}: {stats['success']}/{stats['total']} ({cat_rate:.1f}%)")
        
        print(f"\n📋 DETALHES POR TESTE:")
        print("-" * 80)
        for r in self.results:
            status = "✓" if r.success else "✗"
            print(f"{status} [{r.test_category:18s}] {r.test_name:30s} | "
                  f"Output: {r.output_length:4d} chars | "
                  f"Time: {r.execution_time:5.2f}s")
        
        return {
            "summary": {
                "total_tests": total,
                "successful": success,
                "failed": total - success,
                "success_rate": f"{success_rate:.1f}%",
                "average_time": f"{avg_time:.2f}s",
                "average_output_length": int(avg_output_length)
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
    
    def save_report(self, filename="retriever_node_report.json"):
        report = self.generate_report()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Relatório salvo em: {filename}")
    
    def plot_results(self, filename="retriever_node_plots.png"):
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
        
        # Gráfico 2: Tamanho dos outputs
        test_names = [r.test_name[:15] for r in self.results]
        output_sizes = [r.output_length for r in self.results]
        colors = ['green' if r.success else 'red' for r in self.results]
        
        axes[0, 1].barh(test_names, output_sizes, color=colors, alpha=0.7)
        axes[0, 1].set_xlabel('Tamanho do Output (chars)')
        axes[0, 1].set_title('Tamanho dos Outputs por Teste')
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
    print(" SISTEMA DE TESTES DO RETRIEVER NODE ".center(80, "="))
    print("="*80)
    
    try:
        tester = RetrieverNodeTester()
        tester.run_all_tests()
        tester.generate_report()
        tester.save_report()
        tester.plot_results()
        
        print("\n" + "="*80)
        print(" TESTES CONCLUÍDOS COM SUCESSO ".center(80, "="))
        print("="*80)
        print("\n📁 Arquivos gerados:")
        print("   - retriever_node_report.json")
        print("   - retriever_node_plots.png\n")
        
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()