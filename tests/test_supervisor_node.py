"""
Sistema de Avaliação do SupervisorNode
Arquivo: test_supervisor_node.py

OBJETIVO: Testar se o SupervisorNode toma decisões corretas de planejamento

DESAFIO: Este é o teste MAIS COMPLEXO pois valida raciocínio de alto nível

COMO USAR:
1. Salve este arquivo na pasta tests/
2. Execute: python test_supervisor_node.py
3. Verifique: supervisor_node_report.json e supervisor_node_plots.png
"""

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
from agentai.nodes import SupervisorNode
from agentai.modules.common import AgentState
from langchain_community.chat_models import ChatDeepInfra
from dotenv import load_dotenv


class SupervisorDecisionValidator:
    """Valida decisões do SupervisorNode baseado em regras de negócio"""
    
    VALID_NEXT_STEPS = ["inspect", "imputator", "feature_engineer", "retriever", "plot", "END"]
    
    @staticmethod
    def validate_initial_decision(decision: Dict, main_goal: str) -> Dict:
        """
        Valida decisão inicial do workflow
        Regra: Deve SEMPRE começar com 'inspect' ou 'plot'
        """
        next_step = decision.get("next", "").lower()
        
        if next_step in ["inspect", "plot"]:
            return {
                "passed": True,
                "score": 1.0,
                "reason": f"✓ Início correto com '{next_step}'"
            }
        else:
            return {
                "passed": False,
                "score": 0.0,
                "reason": f"✗ Deveria começar com 'inspect' ou 'plot', não '{next_step}'"
            }
    
    @staticmethod
    def validate_after_missing_detection(
        decision: Dict,
        previous_report: str
    ) -> Dict:
        """
        Valida decisão após detectar dados faltantes
        Regra: Deve ir para 'imputator' se houver missing significativo
        """
        next_step = decision.get("next", "").lower()
        
        # Verificar se o relatório menciona missing data
        has_missing = any(word in previous_report.lower() 
                         for word in ["missing", "nan", "null"])
        
        if not has_missing:
            return {
                "passed": True,
                "score": 1.0,
                "reason": "N/A - Sem dados faltantes detectados"
            }
        
        # Se há missing, deve ir para imputator
        if next_step == "imputator":
            return {
                "passed": True,
                "score": 1.0,
                "reason": "✓ Corretamente delegou para imputator após detectar missing"
            }
        else:
            # Pode ser aceitável ir para END se missing é mínimo
            if "few" in previous_report.lower() or "only" in previous_report.lower():
                return {
                    "passed": True,
                    "score": 0.7,
                    "reason": f"⚠ Foi para '{next_step}' com poucos missing (aceitável)"
                }
            else:
                return {
                    "passed": False,
                    "score": 0.0,
                    "reason": f"✗ Deveria ir para 'imputator', não '{next_step}'"
                }
    
    @staticmethod
    def validate_feature_request(
        decision: Dict,
        main_goal: str,
        previous_msg: str
    ) -> Dict:
        """
        Valida se delega feature engineering corretamente
        Regra: Palavras-chave de feature → deve ir para 'feature_engineer'
        """
        next_step = decision.get("next", "").lower()
        
        # Keywords que indicam necessidade de feature engineering
        feature_keywords = [
            "feature", "rolling", "lag", "average", "create", 
            "transform", "engineer", "derive", "ratio", "difference"
        ]
        
        text_to_check = (main_goal + " " + previous_msg).lower()
        needs_feature = any(keyword in text_to_check for keyword in feature_keywords)
        
        if not needs_feature:
            return {
                "passed": True,
                "score": 1.0,
                "reason": "N/A - Não há solicitação de feature engineering"
            }
        
        if next_step == "feature_engineer":
            return {
                "passed": True,
                "score": 1.0,
                "reason": "✓ Corretamente delegou feature engineering"
            }
        else:
            return {
                "passed": False,
                "score": 0.0,
                "reason": f"✗ Deveria delegar para 'feature_engineer', não '{next_step}'"
            }
    
    @staticmethod
    def validate_error_recovery(
        decision: Dict,
        previous_report: str,
        logs: List[str]
    ) -> Dict:
        """
        Valida recuperação de erros
        Regra: Se há erro, deve usar 'retriever' para buscar solução
        """
        next_step = decision.get("next", "").lower()
        
        # Verificar se há erro no relatório ou logs
        has_error = (
            "error" in previous_report.lower() or
            any("error" in log.lower() for log in logs)
        )
        
        if not has_error:
            return {
                "passed": True,
                "score": 1.0,
                "reason": "N/A - Sem erros detectados"
            }
        
        if next_step == "retriever":
            return {
                "passed": True,
                "score": 1.0,
                "reason": "✓ Corretamente usou retriever para resolver erro"
            }
        elif next_step == "END":
            return {
                "passed": False,
                "score": 0.0,
                "reason": "✗ Finalizou com erro não resolvido"
            }
        else:
            return {
                "passed": True,
                "score": 0.5,
                "reason": f"⚠ Foi para '{next_step}' após erro (pode estar tentando resolver)"
            }
    
    @staticmethod
    def validate_loop_prevention(
        decision: Dict,
        logs: List[str]
    ) -> Dict:
        """
        Valida se não está em loop infinito
        Regra: Não deve repetir o mesmo nó mais de 3 vezes seguidas
        """
        next_step = decision.get("next", "").lower()
        
        # Contar repetições recentes do próximo passo nos logs
        recent_logs = logs[-5:] if len(logs) >= 5 else logs
        log_text = " ".join(recent_logs).lower()
        
        # Contar menções ao próximo nó
        count = log_text.count(next_step)
        
        if count >= 3:
            return {
                "passed": False,
                "score": 0.0,
                "reason": f"✗ Loop detectado: '{next_step}' repetido {count} vezes"
            }
        else:
            return {
                "passed": True,
                "score": 1.0,
                "reason": f"✓ Sem loops ('{next_step}' aparece {count}x)"
            }
    
    @staticmethod
    def validate_json_format(decision: Dict) -> Dict:
        """Valida formato do JSON de decisão"""
        required_fields = ["output", "next", "msg", "is_before_dp"]
        
        missing_fields = [f for f in required_fields if f not in decision]
        
        if missing_fields:
            return {
                "passed": False,
                "score": 0.0,
                "reason": f"✗ Campos faltantes: {missing_fields}"
            }
        
        # Validar 'next'
        next_step = decision.get("next", "").lower()
        if next_step not in SupervisorDecisionValidator.VALID_NEXT_STEPS:
            return {
                "passed": False,
                "score": 0.5,
                "reason": f"⚠ 'next' inválido: '{next_step}'"
            }
        
        # Validar 'is_before_dp'
        is_before_dp = str(decision.get("is_before_dp", "")).lower()
        if is_before_dp not in ["true", "false"]:
            return {
                "passed": False,
                "score": 0.5,
                "reason": f"⚠ 'is_before_dp' deve ser 'true' ou 'false', não '{is_before_dp}'"
            }
        
        return {
            "passed": True,
            "score": 1.0,
            "reason": "✓ JSON válido com todos os campos"
        }


@dataclass
class SupervisorTestResult:
    test_name: str
    test_category: str
    success: bool
    execution_time: float
    scenario_description: str
    decision_made: Dict
    validation_results: Dict
    error_message: str

class SupervisorNodeTester:
    def __init__(self):
        print("\n" + "="*80)
        print("INICIALIZANDO TESTADOR DO SUPERVISOR NODE")
        print("="*80)
        print("\n⚠️  AVISO: Este é o teste mais complexo - valida raciocínio de planejamento")
        
        load_dotenv()
        
        print("\n[1/3] Inicializando LLM...")
        self.llm = ChatDeepInfra(model="Qwen/Qwen2.5-72B-Instruct")
        print("   ✓ LLM inicializado")
        
        print("\n[2/3] Criando dataset inicial...")
        dummy_df = self._create_simple_dataset(n_rows=10)
        dummy_path = "temp_dummy_supervisor.csv"
        dummy_df.to_csv(dummy_path, index=False)
        print(f"   ✓ Dataset salvo em {dummy_path}")
        
        print("\n[3/3] Inicializando WorkflowExecutor...")
        self.executor = WorkflowExecutor(
            csv_path=dummy_path,
            plot_images_path="./test_plots",
            llm=self.llm
        )
        print("   ✓ Executor inicializado")
        
        self.results: List[SupervisorTestResult] = []
        self.validator = SupervisorDecisionValidator()
        
        print("\n✓ Testador pronto!")
    
    def _create_simple_dataset(self, n_rows=100, has_missing=False):
        """Cria dataset simples para testes"""
        timestamps = pd.date_range('2024-01-01', periods=n_rows, freq='h')
        data = {
            'timestamp': timestamps,
            'temperature': 20 + np.random.randn(n_rows) * 2,
            'humidity': 60 + np.random.randn(n_rows) * 5
        }
        df = pd.DataFrame(data)
        
        if has_missing:
            n_missing = int(n_rows * 0.2)
            missing_idx = np.random.choice(n_rows, size=n_missing, replace=False)
            df.loc[missing_idx, 'temperature'] = np.nan
        
        return df
    
    def run_all_tests(self):
        print("\n" + "="*80)
        print("EXECUTANDO BATERIA DE TESTES - SUPERVISOR NODE")
        print("="*80)
        
        # CATEGORIA 1: Decisões Iniciais
        print("\n[CATEGORIA 1/5] Decisões Iniciais do Workflow")
        print("-" * 80)
        self._test_initial_decision()
        
        # CATEGORIA 2: Delegação para Imputação
        print("\n[CATEGORIA 2/5] Delegação para Imputação")
        print("-" * 80)
        self._test_delegate_to_imputator()
        
        # CATEGORIA 3: Delegação para Feature Engineering
        print("\n[CATEGORIA 3/5] Delegação para Feature Engineering")
        print("-" * 80)
        self._test_delegate_to_feature_engineer()
        
        # CATEGORIA 4: Recuperação de Erros
        print("\n[CATEGORIA 4/5] Recuperação de Erros")
        print("-" * 80)
        self._test_error_recovery()
        
        # CATEGORIA 5: Finalização
        print("\n[CATEGORIA 5/5] Decisões de Finalização")
        print("-" * 80)
        self._test_end_decision()
        
        print("\n" + "="*80)
        print("TESTES CONCLUÍDOS")
        print("="*80)
    
    def _run_single_test(
        self,
        test_name: str,
        test_category: str,
        state: AgentState,
        validators: List[callable],
        scenario_description: str
    ):
        print(f"\n   [{test_category}] {test_name}")
        print(f"   Cenário: {scenario_description[:60]}...")
        
        start_time = time.time()
        success = False
        error_message = ""
        decision = {}
        validation_results = {}
        
        try:
            # Criar nó e executar
            node = SupervisorNode(self.executor)
            result = node.execute(state)
            execution_time = time.time() - start_time
            
            # Extrair decisão
            next_step = result.get("next", "")
            msg = result.get("msg", "")
            logs = result.get("logs", [])
            
            # Tentar extrair JSON completo dos logs se necessário
            if logs:
                last_log = logs[-1]
                json_match = re.search(r'\{.*\}', last_log, re.DOTALL)
                if json_match:
                    try:
                        decision = json.loads(json_match.group(0))
                    except:
                        decision = {
                            "next": next_step,
                            "msg": msg,
                            "output": last_log[:100]
                        }
                else:
                    decision = {
                        "next": next_step,
                        "msg": msg,
                        "output": "No JSON in logs"
                    }
            else:
                decision = {
                    "next": next_step,
                    "msg": msg
                }
            
            # Executar validadores
            all_scores = []
            for validator in validators:
                validation = validator(decision, state)
                validation_results[validator.__name__] = validation
                all_scores.append(validation.get("score", 0))
            
            # Calcular sucesso médio
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
            success = avg_score >= 0.7  # 70% de score mínimo
            
            status = "✓ SUCESSO" if success else "✗ FALHA"
            print(f"   {status} - Tempo: {execution_time:.2f}s - Decisão: {decision.get('next', 'N/A')}")
            print(f"      Score médio: {avg_score:.1%}")
            
            # Mostrar validações
            for val_name, val_result in validation_results.items():
                print(f"      {val_name}: {val_result.get('reason', 'N/A')}")
            
        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            error_message = str(e)
            print(f"   ✗ ERRO - {error_message}")
        
        # Salvar resultado
        result = SupervisorTestResult(
            test_name=test_name,
            test_category=test_category,
            success=success,
            execution_time=execution_time,
            scenario_description=scenario_description,
            decision_made=decision,
            validation_results=validation_results,
            error_message=error_message
        )
        self.results.append(result)
    
    
    def _test_initial_decision(self):
        """Teste: primeira decisão do workflow"""
        df = self._create_simple_dataset(n_rows=100)
        self.executor.df = df
        
        state = AgentState(
            msg="Perform exploratory data analysis on this IoT sensor dataset",
            main_goal="Perform exploratory data analysis on this IoT sensor dataset",
            logs=[],
            subagents_report=None
        )
        
        def validator_wrapper(decision, state_param):
            return self.validator.validate_initial_decision(decision, state_param.get("main_goal", ""))
        
        self._run_single_test(
            test_name="initial_workflow_start",
            test_category="InitialDecision",
            state=state,
            validators=[
                validator_wrapper,
                lambda d, s: self.validator.validate_json_format(d)
            ],
            scenario_description="Início do workflow - deve começar com 'inspect' ou 'plot'"
        )
    
    def _test_delegate_to_imputator(self):
        """Teste: delegação após detectar missing data"""
        df = self._create_simple_dataset(n_rows=100, has_missing=True)
        self.executor.df = df
        
        state = AgentState(
            msg="Handle the missing data appropriately",
            main_goal="Perform complete EDA including handling missing data",
            logs=[
                "[Pandas Node]: Analysis complete",
                "[Pandas Node]: Found 20 missing values in temperature column"
            ],
            subagents_report="The dataset has 20 missing values in the temperature column (20%). Missing data treatment is recommended."
        )
        
        def validator_wrapper(decision, state_param):
            return self.validator.validate_after_missing_detection(
                decision,
                state_param.get("subagents_report", "")
            )
        
        self._run_single_test(
            test_name="delegate_to_imputator_after_missing",
            test_category="ImputationDelegation",
            state=state,
            validators=[
                validator_wrapper,
                lambda d, s: self.validator.validate_json_format(d)
            ],
            scenario_description="Após detectar missing data - deve ir para 'imputator'"
        )
    
    def _test_delegate_to_feature_engineer(self):
        """Teste: delegação para feature engineering"""
        df = self._create_simple_dataset(n_rows=100)
        self.executor.df = df
        
        state = AgentState(
            msg="Create a 3-hour rolling average for temperature",
            main_goal="Analyze data and create rolling average features",
            logs=[
                "[Pandas Node]: Initial analysis complete",
                "[Supervisor Node]: Need to create features"
            ],
            subagents_report="Dataset inspection complete. Ready for feature engineering."
        )
        
        def validator_wrapper(decision, state_param):
            return self.validator.validate_feature_request(
                decision,
                state_param.get("main_goal", ""),
                state_param.get("msg", "")
            )
        
        self._run_single_test(
            test_name="delegate_feature_engineering",
            test_category="FeatureDelegation",
            state=state,
            validators=[
                validator_wrapper,
                lambda d, s: self.validator.validate_json_format(d)
            ],
            scenario_description="Solicitação de feature - deve ir para 'feature_engineer'"
        )
    
    def _test_error_recovery(self):
        """Teste: recuperação após erro"""
        df = self._create_simple_dataset(n_rows=100)
        self.executor.df = df
        
        state = AgentState(
            msg="Resolve the error and continue",
            main_goal="Complete the analysis despite errors",
            logs=[
                "[Pandas Node]: Starting analysis",
                "[Pandas Node]: ERROR: NameError - numpy not imported",
                "[Pandas Node]: Analysis failed"
            ],
            subagents_report="ERROR: The pandas agent encountered a NameError. The analysis could not be completed."
        )
        
        def validator_wrapper(decision, state_param):
            return self.validator.validate_error_recovery(
                decision,
                state_param.get("subagents_report", ""),
                state_param.get("logs", [])
            )
        
        self._run_single_test(
            test_name="error_recovery_use_retriever",
            test_category="ErrorRecovery",
            state=state,
            validators=[
                validator_wrapper,
                lambda d, s: self.validator.validate_json_format(d)
            ],
            scenario_description="Após erro - deve usar 'retriever' para buscar solução"
        )
    
    def _test_end_decision(self):
        """Teste: decisão de finalizar workflow"""
        df = self._create_simple_dataset(n_rows=100)
        self.executor.df = df
        
        state = AgentState(
            msg="Finalize the analysis",
            main_goal="Complete exploratory data analysis",
            logs=[
                "[Pandas Node]: Dataset inspected - no issues",
                "[Imputator Node]: No missing data",
                "[Feature Engineer Node]: Features created successfully",
                "[Plotter Node]: Plots generated"
            ],
            subagents_report="All analysis complete. Dataset is clean, features are created, and visualizations are generated."
        )
        
        def validator_end(decision, state_param):
            next_step = decision.get("next", "").lower()
            if next_step == "end":
                return {
                    "passed": True,
                    "score": 1.0,
                    "reason": "✓ Corretamente finalizou após completar análise"
                }
            else:
                return {
                    "passed": False,
                    "score": 0.0,
                    "reason": f"✗ Deveria finalizar (END), não ir para '{next_step}'"
                }
        
        self._run_single_test(
            test_name="end_after_complete_analysis",
            test_category="Finalization",
            state=state,
            validators=[
                validator_end,
                lambda d, s: self.validator.validate_json_format(d)
            ],
            scenario_description="Análise completa - deve finalizar com 'END'"
        )
    

    def generate_report(self):
        print("\n" + "="*80)
        print("RELATÓRIO DE RESULTADOS - SUPERVISOR NODE")
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
        
        # Análise de decisões tomadas
        decisions_made = {}
        for r in self.results:
            decision = r.decision_made.get("next", "unknown")
            decisions_made[decision] = decisions_made.get(decision, 0) + 1
        
        print(f"\n📊 RESUMO GERAL:")
        print(f"   Total de testes: {total}")
        print(f"   Sucessos: {success} ({success_rate:.1f}%)")
        print(f"   Falhas: {total-success} ({100-success_rate:.1f}%)")
        print(f"   Tempo médio: {avg_time:.2f}s")
        
        print(f"\n📊 RESUMO POR CATEGORIA:")
        for cat, stats in categories.items():
            cat_rate = (stats["success"] / stats["total"] * 100)
            print(f"   {cat:20s}: {stats['success']}/{stats['total']} ({cat_rate:.1f}%)")
        
        print(f"\n📊 DECISÕES TOMADAS:")
        for decision, count in decisions_made.items():
            print(f"   {decision.upper():20s}: {count} vezes")
        
        print(f"\n📋 DETALHES POR TESTE:")
        print("-" * 80)
        for r in self.results:
            status = "✓" if r.success else "✗"
            decision = r.decision_made.get("next", "N/A")
            print(f"{status} [{r.test_category:18s}] {r.test_name:35s} | "
                  f"Decisão: {decision:15s} | "
                  f"Time: {r.execution_time:5.2f}s")
            if not r.success and r.error_message:
                print(f"   └─ {r.error_message[:70]}")
        
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
            "decisions_made": decisions_made,
            "details": [asdict(r) for r in self.results]
        }
    
    def save_report(self, filename="supervisor_node_report.json"):
        report = self.generate_report()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Relatório salvo em: {filename}")
    
    def plot_results(self, filename="supervisor_node_plots.png"):
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
        
        # Gráfico 2: Decisões tomadas
        decisions = {}
        for r in self.results:
            decision = r.decision_made.get("next", "unknown")
            decisions[decision] = decisions.get(decision, 0) + 1
        
        if decisions:
            axes[0, 1].bar(decisions.keys(), decisions.values(), color='steelblue', alpha=0.7)
            axes[0, 1].set_ylabel('Frequência')
            axes[0, 1].set_title('Decisões Tomadas pelo Supervisor')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Gráfico 3: Taxa de sucesso por categoria
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
        
        axes[1, 0].barh(cat_names, cat_rates, color='mediumseagreen', alpha=0.7)
        axes[1, 0].set_xlabel('Taxa de Sucesso (%)')
        axes[1, 0].set_title('Sucesso por Categoria')
        axes[1, 0].set_xlim([0, 105])
        axes[1, 0].axvline(x=80, color='r', linestyle='--', label='Meta: 80%')
        axes[1, 0].legend()
        axes[1, 0].invert_yaxis()
        
        # Gráfico 4: Tempo de execução por teste
        test_names = [r.test_name[:20] for r in self.results]
        times = [r.execution_time for r in self.results]
        colors = ['green' if r.success else 'red' for r in self.results]
        
        axes[1, 1].barh(test_names, times, color=colors, alpha=0.7)
        axes[1, 1].set_xlabel('Tempo de Execução (s)')
        axes[1, 1].set_title('Tempo por Teste')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Gráficos salvos em: {filename}")
        plt.close()


def main():
    print("\n" + "="*80)
    print(" SISTEMA DE TESTES DO SUPERVISOR NODE ".center(80, "="))
    print("="*80)
    print("\n⚠️  ATENÇÃO: Este teste valida RACIOCÍNIO DE PLANEJAMENTO")
    print("   É normal ter taxa de sucesso menor que outros nós.\n")
    
    try:
        tester = SupervisorNodeTester()
        tester.run_all_tests()
        tester.generate_report()
        tester.save_report()
        tester.plot_results()
        
        print("\n" + "="*80)
        print(" TESTES CONCLUÍDOS COM SUCESSO ".center(80, "="))
        print("="*80)
        print("\n📁 Arquivos gerados:")
        print("   - supervisor_node_report.json")
        print("   - supervisor_node_plots.png")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()