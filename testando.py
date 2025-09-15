from agentai.rag  import RAG


def main():
    rag = RAG(force_rebuild=True)
    print("Iniciando RAG\n")
    print(rag.retrieve("Poor performance in AutoML generated model in a temporal series context"))
    

main()