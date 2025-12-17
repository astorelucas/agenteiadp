"""
Script Principal de Testes E2E - Menu Interativo
Versão Corrigida e Otimizada
"""

import sys
import os
from pathlib import Path

# Adicionar path do projeto
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent if script_dir.name == 'tests' else script_dir
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
from uuid import uuid4
from dotenv import load_dotenv


def clear_screen():
    """Limpa a tela do terminal"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Imprime cabeçalho do programa"""
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║           FRAMEWORK DE TESTES END-TO-END - AGENTE MULTIAGENTE         ║
║                        Análise de Dados IoT                           ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)


def list_available_datasets(datasets_path: Path):
    """Lista datasets disponíveis no diretório"""
    available = []
    
    if not datasets_path.exists():
        print(f"⚠️  Diretório não encontrado: {datasets_path}")
        return available
    
    for file in datasets_path.glob("*.csv"):
        available.append(file.stem)
    
    return sorted(available)


def display_menu(datasets: list):
    """Exibe menu de seleção de datasets"""
    print("\n" + "─"*75)
    print("DATASETS DISPONÍVEIS:")
    print("─"*75)
    
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset}")
    
    print(f"  {len(datasets) + 1}. TODOS OS DATASETS")
    print("  0. Sair")
    print("─"*75)


def select_datasets(available_datasets: list):
    """Permite seleção de datasets"""
    while True:
        display_menu(available_datasets)
        
        try:
            choice = input(f"\nEscolha uma opção (0-{len(available_datasets) + 1}): ").strip()
            choice_num = int(choice)
            
            if choice_num == 0:
                print("\n👋 Saindo...")
                sys.exit(0)
            elif choice_num == len(available_datasets) + 1:
                return available_datasets
            elif 1 <= choice_num <= len(available_datasets):
                selected = available_datasets[choice_num - 1]
                print(f"\n✓ Selecionado: {selected}")
                return [selected]
            else:
                print(f"\n❌ Opção inválida! Digite um número entre 0 e {len(available_datasets) + 1}")
                input("\nPressione ENTER para continuar...")
        except ValueError:
            print("\n❌ Digite apenas números!")
            input("\nPressione ENTER para continuar...")
        except KeyboardInterrupt:
            print("\n\n👋 Saindo...")
            sys.exit(0)


def ask_inject_missing():
    """Pergunta se deve injetar valores faltantes"""
    print("\n" + "─"*75)
    print("INJEÇÃO DE VALORES FALTANTES:")
    print("─"*75)
    print("  1. SIM - Injetar missing values artificialmente (recomendado para testes)")
    print("  2. NÃO - Usar dados como estão")
    print("─"*75)
    
    while True:
        try:
            choice = input("\nEscolha uma opção (1-2): ").strip()
            
            if choice == "1":
                # Perguntar taxa de missing
                while True:
                    try:
                        rate_str = input("\nTaxa de missing values (5-30%, recomendado 15%): ").strip().replace('%', '')
                        rate = float(rate_str) / 100
                        
                        if 0.05 <= rate <= 0.30:
                            print(f"\n✓ Taxa selecionada: {rate:.1%}")
                            return True, rate
                        else:
                            print("❌ Taxa deve estar entre 5% e 30%")
                    except ValueError:
                        print("❌ Digite um número válido (ex: 15)")
            
            elif choice == "2":
                print("\n✓ Usando dados originais sem injeção")
                return False, 0.0
            
            else:
                print("❌ Opção inválida! Digite 1 ou 2")
        
        except KeyboardInterrupt:
            print("\n\n👋 Saindo...")
            sys.exit(0)


def prepare_dataset(df: pd.DataFrame, inject_missing: bool, missing_rate: float):
    """Prepara dataset com ou sem injeção de missing values"""
    df_original = df.copy()
    
    if inject_missing:
        df_test = df.copy()
        numeric_cols = df_test.select_dtypes(include=[np.number]).columns
        
        print(f"\n  Injetando {missing_rate:.1%} de missing values...")
        
        total_injected = 0
        for col in numeric_cols:
            mask = np.random.random(len(df_test)) < missing_rate
            df_test.loc[mask, col] = np.nan
            total_injected += mask.sum()
        
        print(f"  ✓ {total_injected} valores tornados faltantes")
        
        return df_test, df_original
    else:
        print("  Usando dados originais")
        return df.copy(), df_original


def run_agent_test(csv_path: str, dataset_name: str, prompt: str, output_dir: Path):
    """Executa o agente e retorna resultados"""
    
    print(f"\n{'─'*75}")
    print(f"EXECUTANDO AGENTE: {dataset_name}")
    print(f"{'─'*75}")
    
    # Importar módulos do agente
    try:
        from langchain_community.chat_models import ChatDeepInfra
        from agentai.workflow import WorkflowExecutor
    except ImportError as e:
        print(f"❌ Erro ao importar módulos do agente: {e}")
        print("   Verifique se está executando do diretório correto")
        return None
    
    # Carregar API key
    # Buscar .env no diretório do projeto
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
        load_dotenv()  # Tentar busca padrão
    
    api_key = os.getenv("DEEPINFRA_API_KEY")
    if not api_key:
        print("❌ DEEPINFRA_API_KEY não encontrada no arquivo .env")
        print(f"   Procurou em: {env_path or 'diretórios padrão'}")
        return None
    
    # Criar LLM
    try:
        llm = ChatDeepInfra(model="Qwen/Qwen2.5-72B-Instruct", max_tokens=500)
    except Exception as e:
        print(f"❌ Erro ao criar LLM: {e}")
        return None
    
    # Diretório para plots
    images_path = output_dir / dataset_name / "plots"
    images_path.mkdir(parents=True, exist_ok=True)
    
    # Carregar dataframe original
    try:
        df_original = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        return None

    try:
        # Inicializar executor
        print("  [1/3] Inicializando agente...")
        executor = WorkflowExecutor(
            llm=llm,
            csv_path=csv_path,
            plot_images_path=str(images_path)
        )
        
        # Snapshots
        df_snapshots = {
            'original': df_original.copy(),
            'after_inspection': executor.df.copy(),
            'after_imputation': None,
            'after_feature_eng': None,
            'final': None
        }
        
        # Executar
        print("  [2/3] Executando workflow...")
        thread_id = str(uuid4())
        
        final_state = executor.invoke(
            initial_message=prompt,
            thread_id=thread_id
        )
        
        # Capturar estados finais
        df_snapshots['final'] = executor.df.copy()
        
        # Extrair estados intermediários dos logs
        logs = final_state.get('logs', [])
        df_snapshots = extract_intermediate_states(logs, df_snapshots, executor.df)
        
        print("  [3/3] Agente concluído!")
        print(f"    • Total de logs: {len(logs)}")
        print(f"    • Shape final: {executor.df.shape}")
        
        return {
            'logs': logs,
            'df_original': df_original,
            'df_after_inspection': df_snapshots['after_inspection'],
            'df_after_imputation': df_snapshots['after_imputation'],
            'df_after_feature_eng': df_snapshots['after_feature_eng'],
            'df_final': df_snapshots['final'],
            'summary': final_state.get('summary', ''),
            'images_path': str(images_path),
            'final_state': final_state
        }
        
    except Exception as e:
        print(f"  ❌ Erro ao executar agente: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_intermediate_states(logs: list, snapshots: dict, df_final: pd.DataFrame):
    """Extrai estados intermediários dos logs"""
    
    inspection_done = False
    imputation_done = False
    feature_eng_done = False
    
    for log in logs:
        log_lower = log.lower()
        
        if ('pandas node' in log_lower or '[inspect' in log_lower) and not inspection_done:
            inspection_done = True
            if snapshots['after_inspection'] is None:
                snapshots['after_inspection'] = df_final.copy()
        
        elif 'imputat' in log_lower and not imputation_done:
            imputation_done = True
            if snapshots['after_imputation'] is None:
                snapshots['after_imputation'] = df_final.copy()
        
        elif 'feature' in log_lower and 'engineer' in log_lower and not feature_eng_done:
            feature_eng_done = True
            if snapshots['after_feature_eng'] is None:
                snapshots['after_feature_eng'] = df_final.copy()
    
    # Preencher não capturados
    if snapshots['after_inspection'] is None:
        snapshots['after_inspection'] = snapshots['original'].copy()
    if snapshots['after_imputation'] is None:
        snapshots['after_imputation'] = snapshots['after_inspection'].copy()
    if snapshots['after_feature_eng'] is None:
        snapshots['after_feature_eng'] = snapshots['after_imputation'].copy()
    
    return snapshots


def analyze_results(agent_results: dict, df_with_ground_truth: pd.DataFrame, dataset_name: str):
    """Analisa resultados do agente"""
    
    # Importar do framework
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        from test_framework_e2e import AgentNodeAnalyzer
    except ImportError:
        print("⚠️  test_framework_e2e.py não encontrado. Análise será limitada.")
        return {
            'inspector': {},
            'imputator': {},
            'feature_engineer': {},
            'plotter': {},
            'supervisor': {},
            'automl': {}
        }
    
    print(f"\n{'─'*75}")
    print(f"ANALISANDO RESULTADOS: {dataset_name}")
    print(f"{'─'*75}")
    
    analyzer = AgentNodeAnalyzer()
    
    analysis = {}
    
    print("  [1/6] Analisando Nó Inspector...")
    analysis['inspector'] = analyzer.analyze_inspector_node(
        logs=agent_results['logs'],
        df_original=agent_results['df_original'],
        df_after_inspection=agent_results['df_after_inspection']
    )
    
    print("  [2/6] Analisando Nó Imputator...")
    analysis['imputator'] = analyzer.analyze_imputator_node(
        logs=agent_results['logs'],
        df_before=agent_results['df_after_inspection'],
        df_after=agent_results['df_after_imputation'],
        df_original_complete=df_with_ground_truth
    )
    
    print("  [3/6] Analisando Nó Feature Engineer...")
    analysis['feature_engineer'] = analyzer.analyze_feature_engineer_node(
        logs=agent_results['logs'],
        df_before=agent_results['df_after_imputation'],
        df_after=agent_results['df_after_feature_eng']
    )
    
    print("  [4/6] Analisando Nó Plotter...")
    analysis['plotter'] = analyzer.analyze_plotter_node(
        logs=agent_results['logs'],
        images_path=agent_results['images_path']
    )
    
    print("  [5/6] Analisando Nó Supervisor...")
    analysis['supervisor'] = analyzer.analyze_supervisor_node(
        logs=agent_results['logs']
    )
    
    print("  [6/6] Analisando Nó AutoML...")
    analysis['automl'] = analyzer.analyze_automl_node(
        logs=agent_results['logs']
    )
    
    print("  ✓ Análise concluída!")
    
    return analysis


def generate_report(dataset_name: str, analysis: dict, agent_results: dict, output_dir: Path):
    """Gera relatório detalhado"""
    
    report = f"""
{'='*80}
RELATÓRIO DE TESTE END-TO-END
Dataset: {dataset_name}
Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

## INFORMAÇÕES DO DATASET
{'─'*80}
• Shape Original: {agent_results['df_original'].shape}
• Shape Final: {agent_results['df_final'].shape}
• Colunas Originais: {len(agent_results['df_original'].columns)}
• Colunas Finais: {len(agent_results['df_final'].columns)}

## 1. NÓ INSPETOR
{'─'*80}
"""
    
    inspector = analysis['inspector']
    report += f"• % Faltantes Identificados: "
    if inspector.get('missing_percentage_identified') is not None:
        report += f"{inspector['missing_percentage_identified']:.2f}%\n"
    else:
        report += "Não identificado\n"
    
    report += "\n• Características Extraídas:\n"
    for char, identified in inspector.get('characteristics_extracted', {}).items():
        status = "✓" if identified else "✗"
        report += f"  {status} {char.replace('_', ' ').title()}\n"
    
    report += f"\n• Score de Extração: {inspector.get('characteristics_score', 0):.1%}\n"
    
    if inspector.get('errors'):
        report += "\n• Erros:\n"
        for err in inspector['errors']:
            report += f"  ⚠️  {err}\n"
    
    report += f"\n## 2. NÓ IMPUTATOR\n{'─'*80}\n"
    
    imputator = analysis['imputator']
    report += f"• Método Escolhido: {imputator.get('method_chosen', 'N/A')}\n"
    report += f"• Parâmetros: {imputator.get('parameters', {})}\n"
    report += f"• Valores Imputados: {imputator.get('values_imputed_count', 0)}\n"
    report += f"• Execução Bem-Sucedida: {'✓ Sim' if imputator.get('execution_success') else '✗ Não'}\n"
    
    if imputator.get('error_vs_original'):
        report += "\n• Erro vs Dados Originais:\n"
        report += f"  {'Coluna':<20} {'MAPE (%)':<12} {'RMSE':<12} {'MAE':<12}\n"
        report += f"  {'-'*56}\n"
        for err_info in imputator['error_vs_original']:
            report += f"  {err_info['column']:<20} "
            report += f"{err_info['mape']:>10.2f}% "
            report += f"{err_info['rmse']:>10.4f}  "
            report += f"{err_info['mae']:>10.4f}\n"
        
        if 'avg_mape' in imputator:
            report += f"\n  MAPE Médio Geral: {imputator['avg_mape']:.2f}%\n"
    
    report += f"\n## 3. NÓ FEATURE ENGINEERING\n{'─'*80}\n"
    
    feat_eng = analysis['feature_engineer']
    report += f"• Features Adicionadas: {feat_eng.get('features_count', 0)}\n"
    report += f"• % Faltante Após: {feat_eng.get('missing_percentage_after', 0):.2f}%\n"
    
    if feat_eng.get('features_added'):
        report += f"\n• Lista de Features:\n"
        for i, feat in enumerate(feat_eng['features_added'], 1):
            report += f"  {i}. {feat}\n"
    
    if feat_eng.get('feature_types'):
        report += "\n• Tipos de Features:\n"
        for ftype, features in feat_eng['feature_types'].items():
            report += f"  • {ftype.replace('_', ' ').title()} ({len(features)})\n"
            for feat in features[:5]:
                report += f"    - {feat}\n"
            if len(features) > 5:
                report += f"    ... (+{len(features)-5} mais)\n"
    
    report += f"\n## 4. NÓ PLOTTER\n{'─'*80}\n"
    
    plotter = analysis['plotter']
    report += f"• Total de Plots: {plotter.get('plots_generated', 0)}\n"
    
    if plotter.get('plot_types'):
        report += f"• Tipos: {', '.join(set(plotter['plot_types']))}\n"
    
    if plotter.get('plot_files'):
        report += "\n• Arquivos:\n"
        for pfile in plotter['plot_files']:
            report += f"  - {pfile}\n"
    
    report += f"\n## 5. NÓ SUPERVISOR\n{'─'*80}\n"
    
    supervisor = analysis['supervisor']
    report += f"• Total de Decisões: {len(supervisor.get('decisions', []))}\n"
    
    if supervisor.get('sequence'):
        report += f"\n• Sequência:\n  "
        report += " → ".join(supervisor['sequence']) + "\n"
    
    report += f"\n• Loops: {supervisor.get('loops_detected', 0)}\n"
    report += f"• Total de Erros: {supervisor.get('total_errors', 0)}\n"
    report += f"  - Resolvidos: {supervisor.get('errors_resolved', 0)}\n"
    report += f"  - Não Resolvidos: {supervisor.get('errors_unresolved', 0)}\n"
    
    if supervisor.get('total_errors') > 0:
        rate = supervisor.get('errors_resolved', 0) / supervisor['total_errors']
        report += f"  - Taxa de Resolução: {rate:.1%}\n"
    
    report += f"\n• Qualidade do Planejamento: {supervisor.get('planning_quality', 0):.1%}\n"
    
    if analysis.get('automl', {}).get('executed'):
        report += f"\n## 6. NÓ AUTOML\n{'─'*80}\n"
        automl = analysis['automl']
        report += f"• Executado: ✓ Sim\n"
        report += f"• Modelo: {automl.get('model_selected', 'N/A')}\n"
        
        if automl.get('metrics'):
            report += "\n• Métricas:\n"
            for metric, value in automl['metrics'].items():
                report += f"  - {metric}: {value:.4f}\n"
    
    report += f"\n{'='*80}\n"
    report += f"RESUMO FINAL\n"
    report += f"{'='*80}\n\n"
    report += agent_results.get('summary', 'Nenhum resumo disponível')
    report += f"\n\n{'='*80}\n"
    
    # Salvar
    report_path = output_dir / f"{dataset_name}_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ Relatório salvo: {report_path}")
    
    return report


def print_summary_table(dataset_name: str, analysis: dict):
    """Imprime tabela resumo no terminal"""
    
    print(f"\n{'─'*75}")
    print(f"RESUMO: {dataset_name}")
    print(f"{'─'*75}")
    
    # Calcular scores
    inspector_score = analysis['inspector'].get('characteristics_score', 0)
    imputator_success = 1.0 if analysis['imputator'].get('execution_success') else 0.0
    feat_count = analysis['feature_engineer'].get('features_count', 0)
    plot_count = analysis['plotter'].get('plots_generated', 0)
    supervisor_quality = analysis['supervisor'].get('planning_quality', 0)
    
    # Tratar method_chosen que pode ser None
    method_chosen = analysis['imputator'].get('method_chosen')
    method_str = str(method_chosen) if method_chosen is not None else 'N/A'
    
    print(f"\n{'Nó':<20} {'Métrica':<30} {'Valor':<25}")
    print("-" * 75)
    print(f"{'Inspector':<20} {'Score Extração':<30} {inspector_score:>23.1%}")
    print(f"{'Imputator':<20} {'Método':<30} {method_str:>25}")
    print(f"{'':>20} {'Sucesso':<30} {'Sim' if imputator_success else 'Não':>25}")
    
    if 'avg_mape' in analysis['imputator']:
        print(f"{'':>20} {'MAPE Médio':<30} {analysis['imputator']['avg_mape']:>23.2f}%")
    
    print(f"{'Feature Engineer':<20} {'Features Criadas':<30} {feat_count:>25}")
    print(f"{'Plotter':<20} {'Plots Gerados':<30} {plot_count:>25}")
    print(f"{'Supervisor':<20} {'Qualidade':<30} {supervisor_quality:>23.1%}")
    print(f"{'':>20} {'Erros Resolvidos/Total':<30} {analysis['supervisor'].get('errors_resolved', 0)}/{analysis['supervisor'].get('total_errors', 0):>17}")
    
    # Score geral
    scores = [inspector_score, imputator_success, min(1.0, feat_count/8), 
              min(1.0, plot_count/3), supervisor_quality]
    overall = sum(scores) / len(scores)
    
    print("-" * 75)
    print(f"{'SCORE GERAL':<20} {'':<30} {overall:>23.1%}")
    
    if overall >= 0.85:
        print(f"{'STATUS':<20} {'':<30} {'🎉 EXCELENTE':>25}")
    elif overall >= 0.70:
        print(f"{'STATUS':<20} {'':<30} {'✓ BOM':>25}")
    elif overall >= 0.50:
        print(f"{'STATUS':<20} {'':<30} {'⚠️ REGULAR':>25}")
    else:
        print(f"{'STATUS':<20} {'':<30} {'❌ NECESSITA MELHORIAS':>25}")
    
    print("─" * 75)


def main():
    """Função principal"""
    
    # Detectar diretório do script e projeto
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent if script_dir.name == 'tests' else script_dir
    
    # Definir caminhos relativos ao projeto
    DATASETS_PATH = project_root / "agentai" / "datasets"
    OUTPUT_PATH = project_root / "tests"
    
    # Criar diretório de saída
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Cabeçalho
    print_header()
    
    # Verificar se diretório de datasets existe
    if not DATASETS_PATH.exists():
        print(f"❌ Diretório de datasets não encontrado:")
        print(f"   {DATASETS_PATH.absolute()}")
        print("\nVerifique o caminho e tente novamente.")
        input("\nPressione ENTER para sair...")
        sys.exit(1)
    
    # Listar datasets disponíveis
    print("Procurando datasets...")
    available_datasets = list_available_datasets(DATASETS_PATH)
    
    if not available_datasets:
        print(f"❌ Nenhum dataset (.csv) encontrado em:")
        print(f"   {DATASETS_PATH.absolute()}")
        input("\nPressione ENTER para sair...")
        sys.exit(1)
    
    print(f"✓ {len(available_datasets)} dataset(s) encontrado(s)")
    
    # Selecionar datasets
    selected_datasets = select_datasets(available_datasets)
    
    # Perguntar sobre injeção de missing
    inject_missing, missing_rate = ask_inject_missing()
    
    # Confirmar execução
    print(f"\n{'='*75}")
    print("CONFIGURAÇÃO DO TESTE:")
    print(f"{'='*75}")
    print(f"  Datasets: {', '.join(selected_datasets)}")
    print(f"  Injetar Missing: {'SIM' if inject_missing else 'NÃO'}")
    if inject_missing:
        print(f"  Taxa: {missing_rate:.1%}")
    print(f"  Saída: {OUTPUT_PATH.absolute()}")
    print(f"{'='*75}")
    
    confirm = input("\nDeseja continuar? (S/N): ").strip().upper()
    if confirm != 'S':
        print("\n👋 Teste cancelado")
        sys.exit(0)
    
    # Prompt padrão
    PROMPT = """
Perform a comprehensive exploratory data analysis on this time series dataset:

1. INSPECTION: Analyze structure, types, shape, and identify missing values with exact percentages
2. IMPUTATION: Handle missing values using the most appropriate method
3. FEATURE ENGINEERING: Create relevant time series features (rolling windows, lags, temporal)
4. VISUALIZATION: Generate plots (time series, correlations, distributions)
5. SUMMARY: Provide comprehensive findings

Be thorough and systematic.
"""
    
    # Executar testes
    all_results = {}
    
    for i, dataset_name in enumerate(selected_datasets, 1):
        print(f"\n\n{'='*75}")
        print(f"TESTE {i}/{len(selected_datasets)}: {dataset_name}")
        print(f"{'='*75}")
        
        # Carregar dataset
        dataset_path = DATASETS_PATH / f"{dataset_name}.csv"
        print(f"\nCarregando dataset...")
        
        try:
            df = pd.read_csv(dataset_path)
            print(f"✓ Carregado: {df.shape}")
        except Exception as e:
            print(f"❌ Erro ao carregar dataset: {e}")
            continue
        
        # Preparar dataset
        df_test, df_original = prepare_dataset(df, inject_missing, missing_rate)
        
        # Salvar datasets
        test_csv = OUTPUT_PATH / f"{dataset_name}_test.csv"
        original_csv = OUTPUT_PATH / f"{dataset_name}_original.csv"
        
        df_test.to_csv(test_csv, index=False)
        df_original.to_csv(original_csv, index=False)
        
        print(f"  ✓ Salvo em: {test_csv}")
        
        # Executar agente
        agent_results = run_agent_test(
            csv_path=str(test_csv),
            dataset_name=dataset_name,
            prompt=PROMPT,
            output_dir=OUTPUT_PATH
        )
        
        if agent_results is None:
            print(f"\n❌ Falha ao executar {dataset_name}")
            continue
        
        # Analisar
        analysis = analyze_results(agent_results, df_original, dataset_name)
        
        # Gerar relatório
        generate_report(dataset_name, analysis, agent_results, OUTPUT_PATH)
        
        # Mostrar resumo
        print_summary_table(dataset_name, analysis)
        
        # Salvar resultados
        all_results[dataset_name] = analysis
        
        if i < len(selected_datasets):
            input("\nPressione ENTER para continuar com o próximo dataset...")
    
    # Resumo final
    if len(all_results) > 1:
        print(f"\n\n{'='*75}")
        print("RESUMO COMPARATIVO")
        print(f"{'='*75}")
        
        print(f"\n{'Dataset':<15} {'Inspector':<12} {'Imputator':<12} {'Feat.Eng':<12} {'Plotter':<12} {'Supervisor':<12}")
        print("-" * 75)
        
        for dataset_name, analysis in all_results.items():
            inspector_score = analysis['inspector'].get('characteristics_score', 0)
            imputator_success = 1.0 if analysis['imputator'].get('execution_success') else 0.0
            feat_score = min(1.0, analysis['feature_engineer'].get('features_count', 0) / 8)
            plot_score = min(1.0, analysis['plotter'].get('plots_generated', 0) / 3)
            supervisor_score = analysis['supervisor'].get('planning_quality', 0)
            
            print(f"{dataset_name:<15} {inspector_score:>10.1%} {imputator_success:>11.1%} {feat_score:>11.1%} {plot_score:>11.1%} {supervisor_score:>11.1%}")
    
    # Finalização
    print(f"\n\n{'='*75}")
    print("TESTES CONCLUÍDOS!")
    print(f"{'='*75}")
    print(f"\nResultados salvos em: {OUTPUT_PATH.absolute()}")
    print("\nArquivos gerados:")
    print(f"  • *_test.csv - Datasets usados no teste")
    print(f"  • *_original.csv - Datasets originais (para comparação)")
    print(f"  • *_report.txt - Relatórios detalhados")
    print(f"  • */plots/ - Visualizações geradas pelo agente")
    
    input("\nPressione ENTER para sair...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Teste interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        input("\nPressione ENTER para sair...")