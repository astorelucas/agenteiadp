# OPA – Observe, Preprocess, and Act

Project developed by the Agents-of-Future initiative of FutureLab (DCC-UFMG) in partnership with KUNUMI S/A.

# OPA Architecture

>  <p align="left">
>  <img src="https://github.com/astorelucas/agenteiadp/blob/main/agentai/workflow_graph.png?raw=true" alt="Arquitetura" width="500"/>
>  </p> 

## Structure

```plaintext

agenteiadp/
├── agentai/                # Módulos principais
|   ├── FAISS_DB/           # Datasets
|   ├── datasets/           # Datasets
|   ├── modules/            # Utilitários
│   ├── __init__.py
│   ├── agents.py           # Agente inteligente principal
│   ├── base_rag.txt        # RAG
│   ├── nodes.py            # Nós
│   ├── rag.py              # RAG
│   ├── tools.py            # Ferramentas dos agentes
│   └── workflow.py         # Grafo de orquestração
│   ├── workflow_graph.png  # Arquitetura
├── help/                   # Algumas orientações
├── notebooks/              # Notebooks testes
│   └── datasets/           # Datasets utilizados nos notebooks
├── app.py                  # Streamlit
├── main.py                 # Executer
├── requirements.txt        # Dependências do projeto
└── README.md               #  Este arquivo
```

---

## Authors

- Gabriel de Souza Gomes
- Lucas Malacarne Astore
- Luisa Marques Laboissiere
- Thiago Lucas de Oliveira


---
