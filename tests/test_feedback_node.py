import time
import json
import os
import sys
from typing import Dict, List
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
from unittest.mock import Mock, patch

# Adiciona o diretório do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agentai.workflow import WorkflowExecutor
from agentai.nodes import FeedbackNode
from agentai.modules.common import AgentState
from langchain_community.chat_models import ChatDeepInfra
from dotenv import load_dotenv

class MockRAG:
    """Mock do sistema RAG para testes"""
    def __init__(self):
        self.stored_insights = []
        self.vectorstore = Mock()  # Mock do vectorstore
        self.retriever = Mock()
    
    def store(self, text: str):
        """Simula armazenamento de insight"""
        self.stored_insights.append(text)
        print(f"[MockRAG] Insight armazenado: {text[:50]}...")
    
    def retrieve(self, query: str):
        """Simula recuperação de informação"""
        return "Mock retrieval result"

class FeedbackValidator:
    """Valida decisões do FeedbackNode"""
    
    @staticmethod
    def validate_json_format(raw_output: str) -> Dict:
        """Valida se o output é JSON válido com campos corretos"""
        
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        
        if not json_match:
            return {
                "passed": False,
                "score": 0.0,
                "reason": "✗ Nenhum JSON encontrado no output",
                "decision": None
            }
        
        try:
            decision = json.loads(json_match.group(0))
            
            has_store = "store" in decision
            has_insight = "insight" in decision
            
            if not has_store:
                return {
                    "passed": False,
                    "score": 0.0,
                    "reason": "✗ Campo 'store' faltando",
                    "decision": decision
                }
            
            if not has_insight:
                return {
                    "passed": False,
                    "score": 0.5,
                    "reason": "⚠ Campo 'insight' faltando",
                    "decision": decision
                }
            
            if not isinstance(decision["store"], bool):
                return {
                    "passed": False,
                    "score": 0.5,
                    "reason": f"⚠ 'store' deve ser booleano, não {type(decision['store'])}",
                    "decision": decision
                }
            
            return {
                "passed": True,
                "score": 1.0,
                "reason": "✓ JSON válido com todos os campos",
                "decision": decision
            }
            
        except json.JSONDecodeError as e:
            return {
                "passed": False,
                "score": 0.0,
                "reason": f"✗ JSON inválido: {e}",
                "decision": None
            }
    
    @staticmethod
    def validate_decision_logic(
        logs: List[str],
        summary: str,
        decision: Dict,
        expected_store: bool = None
    ) -> Dict:
        """Valida se a decisão faz sentido baseado nos logs"""
        
        if decision is None:
            return {
                "passed": False,
                "score": 0.0,
                "reason": "Decisão não disponível",
                "store_decision": None,
                "insight_provided": False
            }
        
        store_decision = decision.get("store", False)
        insight = decision.get("insight", "")
        
        has_error = any("error" in log.lower() for log in logs)
        has_success = any("success" in log.lower() for log in logs)
        is_complex = len(logs) > 10
        
        if expected_store is not None:
            if store_decision == expected_store:
                return {
                    "passed": True,
                    "score": 1.0,
                    "reason": f"✓ Decisão correta: store={store_decision}",
                    "store_decision": store_decision,
                    "insight_provided": bool(insight)
                }
            else:
                return {
                    "passed": False,
                    "score": 0.0,
                    "reason": f"✗ Esperava store={expected_store}, obteve {store_decision}",
                    "store_decision": store_decision,
                    "insight_provided": bool(insight)
                }
        
        if has_error and has_success:
            if store_decision:
                return {
                    "passed": True,
                    "score": 1.0,
                    "reason": "✓ Corretamente decidiu armazenar (erro resolvido)",
                    "store_decision": store_decision,
                    "insight_provided": bool(insight)
                }
            else:
                return {
                    "passed": False,
                    "score": 0.5,
                    "reason": "⚠ Deveria armazenar aprendizado (erro foi resolvido)",
                    "store_decision": store_decision,
                    "insight_provided": bool(insight)
                }
        
        if not is_complex and not has_error:
            if not store_decision:
                return {
                    "passed": True,
                    "score": 1.0,
                    "reason": "✓ Corretamente decidiu NÃO armazenar (workflow trivial)",
                    "store_decision": store_decision,
                    "insight_provided": bool(insight)
                }
            else:
                return {
                    "passed": False,
                    "score": 0.5,
                    "reason": "⚠ Workflow trivial, não há muito a aprender",
                    "store_decision": store_decision,
                    "insight_provided": bool(insight)
                }
        
        return {
            "passed": True,
            "score": 0.8,
            "reason": "⚠ Cenário ambíguo - decisão aceitável",
            "store_decision": store_decision,
            "insight_provided": bool(insight)
        }
    
    @staticmethod
    def validate_insight_quality(decision: Dict) -> Dict:
        """Valida qualidade do insight se store=True"""
        
        if decision is None:
            return {
                "passed": False,
                "score": 0.0,
                "reason": "Decisão não disponível",
                "has_insight": False
            }
        
        store_decision = decision.get("store", False)
        insight = decision.get("insight", "").strip()
        
        if not store_decision:
            return {
                "passed": True,
                "score": 1.0,
                "reason": "✓ Não precisa de insight (store=False)",
                "has_insight": False
            }
        
        if not insight:
            return {
                "passed": False,
                "score": 0.0,
                "reason": "✗ store=True mas insight vazio",
                "has_insight": False
            }
        
        min_length = 20
        if len(insight) < min_length:
            return {
                "passed": False,
                "score": 0.3,
                "reason": f"⚠ Insight muito curto ({len(insight)} chars)",
                "has_insight": True,
                "insight_length": len(insight)
            }
        
        is_generic = any(phrase in insight.lower() for phrase in [
            "workflow completed",
            "task done",
            "finished successfully",
            "no issues"
        ])
        
        if is_generic:
            return {
                "passed": False,
                "score": 0.5,
                "reason": "⚠ Insight muito genérico",
                "has_insight": True,
                "insight_length": len(insight)
            }
        
        return {
            "passed": True,
            "score": 1.0,
            "reason": "✓ Insight de boa qualidade",
            "has_insight": True,
            "insight_length": len(insight)
        }

# CASOS DE TESTE

@dataclass
class TestCase:
    name: str
    logs: List[str]
    summary: str
    expected_store: bool = None
    description: str = ""

def get_test_cases() -> List[TestCase]:
    """Define os casos de teste"""
    
    return [
        # Caso 1: Erro resolvido - DEVE armazenar
        TestCase(
            name="erro_resolvido",
            logs=[
                "[Supervisor] Starting analysis",
                "[Pandas Node] Error: Column 'temperature' not found",
                "[Retriever] Found solution in knowledge base",
                "[Pandas Node] Successfully analyzed temperature data",
                "[Supervisor] Task completed successfully"
            ],
            summary="The agent encountered a column name error but successfully resolved it using the retriever.",
            expected_store=True,
            description="Cenário onde um erro foi encontrado e resolvido"
        ),
        
        # Caso 2: Workflow trivial - NÃO deve armazenar
        TestCase(
            name="workflow_trivial",
            logs=[
                "[Supervisor] Starting simple inspection",
                "[Pandas Node] Dataset has 100 rows and 5 columns",
                "[Supervisor] Task completed"
            ],
            summary="Simple dataset inspection completed without issues.",
            expected_store=False,
            description="Workflow simples sem aprendizados significativos"
        ),
        
        # Caso 3: Feature engineering bem-sucedida - DEVE armazenar
        TestCase(
            name="feature_engineering_sucesso",
            logs=[
                "[Supervisor] Creating new features",
                "[FeatureEngineeringNode] Created rolling average feature",
                "[FeatureEngineeringNode] Added lag features",
                "[Pandas Node] New features improved analysis",
                "[Supervisor] Features created successfully"
            ],
            summary="Successfully created rolling average and lag features that improved the analysis.",
            expected_store=True,
            description="Criação bem-sucedida de features úteis"
        ),
        
        # Caso 4: Múltiplos erros sem resolução - NÃO deve armazenar
        TestCase(
            name="erros_nao_resolvidos",
            logs=[
                "[Supervisor] Starting imputation",
                "[ImputatorNode] Error: Too many missing values",
                "[ImputatorNode] Error: KNN imputation failed",
                "[ImputatorNode] Error: MICE imputation failed",
                "[Supervisor] Unable to complete imputation"
            ],
            summary="Multiple imputation attempts failed without finding a solution.",
            expected_store=False,
            description="Erros repetidos sem resolução não geram aprendizado"
        ),
        
        # Caso 5: Estratégia de imputação eficaz - DEVE armazenar
        TestCase(
            name="imputacao_eficaz",
            logs=[
                "[Supervisor] Analyzing missing values",
                "[ImputatorNode] KNN imputation performed poorly",
                "[ImputatorNode] MICE imputation with n_estimators=10 worked well",
                "[Pandas Node] Data quality improved significantly",
                "[Supervisor] Imputation completed successfully"
            ],
            summary="MICE imputation with 10 estimators proved effective for this time-series dataset.",
            expected_store=True,
            description="Descoberta de método de imputação eficaz"
        ),
        
        # Caso 6: Workflow complexo mas rotineiro - Ambíguo
        TestCase(
            name="workflow_complexo_rotineiro",
            logs=[
                "[Supervisor] Starting comprehensive EDA",
                "[Pandas Node] Analyzed 50 columns",
                "[PlotterNode] Created 10 visualizations",
                "[Pandas Node] Generated correlation matrix",
                "[Supervisor] EDA completed"
            ] + [f"[Pandas Node] Processed batch {i}" for i in range(15)],
            summary="Comprehensive EDA completed with standard procedures.",
            expected_store=None,  # Ambíguo
            description="Workflow complexo mas usando procedimentos padrão"
        ),
        
        # Caso 7: Descoberta de padrão nos dados - DEVE armazenar
        TestCase(
            name="descoberta_padrao",
            logs=[
                "[Supervisor] Analyzing temporal patterns",
                "[Pandas Node] Detected strong 24-hour cyclical pattern",
                "[FeatureEngineeringNode] Created hour-of-day feature",
                "[Pandas Node] Hour feature significantly improved predictions",
                "[Supervisor] Pattern discovery successful"
            ],
            summary="Discovered a strong 24-hour cyclical pattern that improved model performance.",
            expected_store=True,
            description="Descoberta de padrão importante nos dados"
        ),
        
        # Caso 8: Tentativa e erro até sucesso - DEVE armazenar
        TestCase(
            name="tentativa_erro_sucesso",
            logs=[
                "[Supervisor] Creating visualization",
                "[PlotterNode] Error: Time column not datetime format",
                "[Retriever] Found conversion solution",
                "[FeatureEngineeringNode] Converted column to datetime",
                "[PlotterNode] Successfully created time series plot",
                "[Supervisor] Visualization completed"
            ],
            summary="Resolved datetime conversion issue by consulting knowledge base.",
            expected_store=True,
            description="Processo de tentativa e erro que levou ao sucesso"
        )
    ]


class TestRunner:
    def __init__(self, llm):
        self.llm = llm
        self.results = []
        
    def run_test(self, test_case: TestCase) -> Dict:
        """Executa um único teste"""
        
        print(f"\n{'='*70}")
        print(f"TESTE: {test_case.name}")
        print(f"Descrição: {test_case.description}")
        print(f"{'='*70}")
        
        # Mock do executor mínimo
        class MockExecutor:
            def __init__(self, llm):
                self.llm = llm
        
        executor = MockExecutor(self.llm)
        
        # Criar o FeedbackNode com RAG mockado
        with patch('agentai.nodes.RAG', return_value=MockRAG()):
            feedback_node = FeedbackNode(executor)
            
            # Criar estado simulado
            state = {
                "logs": test_case.logs,
                "summary": test_case.summary
            }
            
            # Executar o nó
            start_time = time.time()
            try:
                result = feedback_node.execute(state)
                execution_time = time.time() - start_time
                
                # Extrair output do agente
                agent_output = result.get("subagents_report", "")
                
            except Exception as e:
                execution_time = time.time() - start_time
                agent_output = f"Error during execution: {str(e)}"
                print(f"\n⚠ ERRO NA EXECUÇÃO: {e}\n")
        
        print(f"\nOutput do Agente:\n{agent_output}\n")
        
        # Validação 1: Formato JSON
        json_validation = FeedbackValidator.validate_json_format(agent_output)
        print(f"✓ Validação JSON: {json_validation['reason']}")
        
        # Validação 2: Lógica da decisão
        logic_validation = FeedbackValidator.validate_decision_logic(
            test_case.logs,
            test_case.summary,
            json_validation.get("decision"),
            test_case.expected_store
        )
        print(f"✓ Validação Lógica: {logic_validation['reason']}")
        
        # Validação 3: Qualidade do insight
        insight_validation = FeedbackValidator.validate_insight_quality(
            json_validation.get("decision")
        )
        print(f"✓ Validação Insight: {insight_validation['reason']}")
        
        # Calcular score final
        final_score = (
            json_validation["score"] * 0.3 +
            logic_validation["score"] * 0.5 +
            insight_validation["score"] * 0.2
        )
        
        passed = json_validation["passed"] and logic_validation["passed"] and insight_validation["passed"]
        
        test_result = {
            "test_name": test_case.name,
            "passed": passed,
            "score": final_score,
            "execution_time": execution_time,
            "json_validation": json_validation,
            "logic_validation": logic_validation,
            "insight_validation": insight_validation,
            "agent_output": agent_output,
            "expected_store": test_case.expected_store
        }
        
        print(f"\n{'─'*70}")
        print(f"RESULTADO: {'✓ PASSOU' if passed else '✗ FALHOU'}")
        print(f"Score Final: {final_score:.2f}/1.00")
        print(f"Tempo: {execution_time:.2f}s")
        
        self.results.append(test_result)
        return test_result
    
    def run_all_tests(self, test_cases: List[TestCase]):
        """Executa todos os testes"""
        
        print("\n" + "="*70)
        print("INICIANDO BATERIA DE TESTES DO FEEDBACKNODE")
        print("="*70)
        
        for test_case in test_cases:
            try:
                self.run_test(test_case)
            except Exception as e:
                print(f"\n✗ ERRO NO TESTE {test_case.name}: {e}")
                import traceback
                traceback.print_exc()
                self.results.append({
                    "test_name": test_case.name,
                    "passed": False,
                    "score": 0.0,
                    "error": str(e)
                })
        
        self.print_summary()
        self.generate_report()
    
    def print_summary(self):
        """Imprime resumo dos testes"""
        
        print("\n" + "="*70)
        print("RESUMO DOS TESTES")
        print("="*70)
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get("passed", False))
        avg_score = sum(r.get("score", 0) for r in self.results) / total if total > 0 else 0
        
        print(f"\nTotal de Testes: {total}")
        print(f"Aprovados: {passed} ({passed/total*100:.1f}%)")
        print(f"Reprovados: {total - passed}")
        print(f"Score Médio: {avg_score:.2f}/1.00")
        
        print("\nDetalhamento por Teste:")
        for result in self.results:
            status = "✓" if result.get("passed") else "✗"
            score = result.get("score", 0)
            expected = result.get("expected_store")
            exp_str = f" (esperado: store={expected})" if expected is not None else ""
            print(f"  {status} {result['test_name']}: {score:.2f}/1.00{exp_str}")
    
    def generate_report(self):
        """Gera relatório visual"""
        
        # Criar DataFrame com resultados
        df = pd.DataFrame([
            {
                "Teste": r["test_name"],
                "Aprovado": r.get("passed", False),
                "Score": r.get("score", 0),
                "Tempo": r.get("execution_time", 0)
            }
            for r in self.results
        ])
        
        # Criar visualizações
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Score por teste
        colors = ['green' if p else 'red' for p in df["Aprovado"]]
        axes[0, 0].barh(df["Teste"], df["Score"], color=colors)
        axes[0, 0].set_xlabel("Score")
        axes[0, 0].set_title("Score por Teste")
        axes[0, 0].axvline(x=0.7, color='orange', linestyle='--', label='Threshold')
        axes[0, 0].legend()
        
        # 2. Taxa de aprovação
        passed_count = df["Aprovado"].sum()
        failed_count = len(df) - passed_count
        axes[0, 1].pie([passed_count, failed_count], labels=["Aprovado", "Reprovado"],
                       autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
        axes[0, 1].set_title("Taxa de Aprovação")
        
        # 3. Tempo de execução
        axes[1, 0].bar(range(len(df)), df["Tempo"], color='steelblue')
        axes[1, 0].set_ylabel("Tempo (s)")
        axes[1, 0].set_title("Tempo de Execução por Teste")
        axes[1, 0].set_xticks(range(len(df)))
        axes[1, 0].set_xticklabels(df["Teste"], rotation=45, ha='right')
        
        # 4. Distribuição de scores
        axes[1, 1].hist(df["Score"], bins=10, edgecolor='black', color='skyblue')
        axes[1, 1].set_xlabel("Score")
        axes[1, 1].set_ylabel("Frequência")
        axes[1, 1].set_title("Distribuição de Scores")
        axes[1, 1].axvline(x=0.7, color='orange', linestyle='--', label='Threshold')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig("feedback_node_test_report.png", dpi=300, bbox_inches='tight')
        print("\n✓ Relatório visual salvo em: feedback_node_test_report.png")
        
        # Salvar resultados em JSON
        with open("feedback_node_test_results.json", "w", encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print("✓ Resultados detalhados salvos em: feedback_node_test_results.json")


def main():
    load_dotenv()
    
    # Configurar LLM
    llm = ChatDeepInfra(model="Qwen/Qwen2.5-72B-Instruct")
    
    # Obter casos de teste
    test_cases = get_test_cases()
    
    # Executar testes
    runner = TestRunner(llm)
    runner.run_all_tests(test_cases)
    
    print("\n" + "="*70)
    print("TESTES CONCLUÍDOS!")
    print("="*70)

if __name__ == "__main__":
    main()