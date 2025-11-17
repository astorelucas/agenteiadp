import streamlit as st
import sys
import os
from uuid import uuid4
import pandas as pd
import tempfile

# Adiciona o diretório do projeto ao path para encontrar o pacote 'agentai'
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Importa a classe principal do agente
from agentai.workflow import WorkflowExecutor
from langchain_community.chat_models import ChatDeepInfra

st.set_page_config(page_title="Agente de Análise de Dados", layout="wide")
st.title("🤖 Agente de Pré-processamento de Dados")
st.markdown("Faça o upload de um arquivo CSV e dê instruções em linguagem natural para o agente analisar e transformar seus dados.")

if 'executor' not in st.session_state:
    st.session_state.executor = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_modificado' not in st.session_state:
    st.session_state.df_modificado = None

#Sidebar
with st.sidebar:
    st.header("1. Carregar Dados")
    uploaded_file = st.file_uploader(
        "Escolha um arquivo CSV",
        type="csv"
    )

    
    if uploaded_file is not None:
        if st.button("Carregar e Iniciar Agente"):
            with st.spinner("Lendo o arquivo e inicializando o agente..."):
                try:
                    # Salvar CSV em arquivo temporário
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        csv_path = tmp_file.name

                    # Ler CSV para exibição no Streamlit
                    df = pd.read_csv(csv_path)

                    st.session_state.df_original = df.copy()
                    st.session_state.df_modificado = None  # Reseta o df modificado

                    # Criar instância do LLM (mesmo modelo usado no main.py)
                    llm = ChatDeepInfra(model="Qwen/Qwen2.5-72B-Instruct")

                    base_dir = os.path.dirname(__file__)
                    images_path = os.path.join(base_dir, "agentai", "images", "plots")

                    # Inicializa o WorkflowExecutor com o caminho do CSV
                    st.session_state.executor = WorkflowExecutor(
                        llm=llm,
                        plot_images_path=images_path,
                        csv_path=csv_path
                    )
                    
                    st.success("Agente pronto para receber instruções!")
                except Exception as e:
                    st.error(f"Erro ao processar o arquivo: {e}")
                    st.session_state.executor = None

if st.session_state.executor is None:
    st.info("Por favor, carregue um arquivo CSV na barra lateral para começar.")
else:
    # Exibe o DataFrame original
    st.header("Visão Geral dos Dados Carregados")
    st.dataframe(st.session_state.df_original)

    st.header("2. Instruções para o Agente")
    prompt_usuario = st.text_area(
        "Descreva a tarefa que você quer que o agente execute:",
        height=150,
        placeholder="Ex: Crie uma feature de média móvel de 3 horas para a temperatura e depois um resumo dos dados."
    )

   
    if st.button("Executar Agente"):
        if prompt_usuario:
            thread_id = str(uuid4())
            
            with st.spinner("O agente está trabalhando... Por favor, aguarde."):
                try:
                    # Pega a instância do executor da sessão
                    executor_instance = st.session_state.executor
                    
                    # Executa o agente
                    final_state = executor_instance.invoke(
                        initial_message=prompt_usuario, 
                        thread_id=thread_id
                    )

                    
                    st.session_state.df_modificado = executor_instance.df.copy()
                    
                    st.session_state.last_result = final_state
                    
                    st.success("O agente concluiu a tarefa!")

                except Exception as e:
                    st.error(f"Ocorreu um erro durante a execução do agente: {e}")
                    # Limpa o resultado anterior em caso de erro
                    st.session_state.df_modificado = None 
        else:
            st.warning("Por favor, insira uma instrução para o agente.")

    if st.session_state.df_modificado is not None:
        st.header("3. DataFrame Processado")
        st.dataframe(st.session_state.df_modificado)

       
        if 'final_state' in locals():
             with st.expander("Ver Logs da Execução"):
                log_formatado = "\n".join([f"- {log}" for log in final_state.get("logs", [])])
                st.markdown(f"```\n{log_formatado}\n```")
                
        if "last_result" in st.session_state:
            st.header("📊 Visualizações Geradas")

            images_path = os.path.join(os.path.dirname(__file__), "agentai", "images", "plots")
            if os.path.exists(images_path):
                for root, dirs, files in os.walk(images_path):
                    for file in files:
                        if file.endswith(".png"):
                            st.image(os.path.join(root, file), caption=file, width="content")


        else:
            st.info("Nenhuma imagem foi gerada.")

        with st.expander("📝 Logs da Execução", expanded=False):
            logs = st.session_state.last_result.get("logs", [])
            if logs:
                log_formatado = "\n".join([f"- {log}" for log in logs])
                st.markdown(f"```\n{log_formatado}\n```")
            else:
                st.info("Nenhum log registrado.")