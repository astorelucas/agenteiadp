import streamlit as st
import sys
import os
from uuid import uuid4

# Adiciona o diretório do projeto ao path para encontrar o pacote 'agentai'
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Importa a classe principal do seu agente
from agentai.workflow import WorkflowExecutor

#Configuração da Página do Streamlit
st.set_page_config(page_title="Agente de Análise de Dados", layout="wide")
st.title("🤖 Agente de Pré-processamento de Dados")
st.markdown("""
Use esta interface para dar instruções ao seu agente de IA. 
""")

# Inicialização do Agente
def load_agent_executor():
    csv_path = "agentai/datasets/test.csv"  
    try:
        executor = WorkflowExecutor(csv_path=csv_path)
        return executor
    except Exception as e:
        st.error(f"Erro ao carregar o agente: {e}")
        return None

executor = load_agent_executor()

if executor:
    #Interface do Usuário
    st.header("Dê uma instrução para o Agente")

    # Caixa de texto para o usuário inserir o prompt
    prompt_usuario = st.text_area(
        "Descreva a tarefa que você quer que o agente execute:",
        height=150,
        placeholder="Ex: Crie uma feature de média móvel de 3 horas para a temperatura e depois um resumo dos dados."
    )

    # Botão para iniciar a execução
    if st.button("Executar Agente"):
        if prompt_usuario:
            # Gera um ID único para cada execução
            thread_id = str(uuid4())
            
            # Mostra uma mensagem de "rodando"
            with st.spinner("O agente está trabalhando... Por favor, aguarde."):
                try:
                    
                    final_state = executor.invoke(initial_message=prompt_usuario, thread_id=thread_id)

                    # Exibição dos Resultados
                    st.success("O agente concluiu a tarefa!")
                    
                    st.subheader("Resultado Final")
                    st.write(final_state) # Exibe o dicionário de estado final

                    st.subheader("Logs da Execução")
                    # Formata os logs para melhor visualização
                    log_formatado = "\n".join([f"- {log}" for log in final_state.get("logs", [])])
                    st.markdown(f"```\n{log_formatado}\n```")

                except Exception as e:
                    st.error(f"Ocorreu um erro durante a execução do agente: {e}")
        else:
            st.warning("Por favor, insira uma instrução para o agente.")

else:
    st.error("A aplicação não pôde ser iniciada porque o agente não foi carregado.")

