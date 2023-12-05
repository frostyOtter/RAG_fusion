import argparse
from utils import read_pdf_from_link, get_vector_store_index
from adv_retriever import generate_queries, FusionRetriever
from llama_index.retrievers import BM25Retriever
from llama_index.query_engine import RetrieverQueryEngine

def config():
    parser = argparse.ArgumentParser(description='Run this file to chat with your pdf file')
    parser.add_argument('--model', type=str, default="sentence-transformers/all-mpnet-base-v2",
                                    help='path to huggingface model')
    parser.add_argument('--local_path', type=str, default="/content/paper.pdf",
                                    help='path to your pdf')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Init input args
    args = config()
    # Paste pdf link
    link_to_pdf = input("Paste your link to pdf file: ")
    # Read pdf file from given link
    my_documents = read_pdf_from_link(pdf_path = link_to_pdf, local_path=args.local_path)
    # Init Vector Store Index
    index = get_vector_store_index(my_documents)

    # Init custom prompt
    query_str = "How do the models developed in this work compare to open-source chat models based on the benchmarks tested?"
    query_gen_prompt_str = (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
    )

    # Init LLM 
    llm = args.model
    # Get queries from LLM
    queries = generate_queries(llm, query_gen_prompt_str, query_str, num_queries=4)
    ## vector retriever
    vector_retriever = index.as_retriever(similarity_top_k=2)
    bm25_retriever = BM25Retriever.from_defaults(
                                docstore=index.docstore,
                                similarity_top_k=2
                            )

    fusion_retriever = FusionRetriever(
        llm, [vector_retriever, bm25_retriever], similarity_top_k=2
    )

    query_engine = RetrieverQueryEngine(fusion_retriever)
    response = query_engine.query(query_str)
    print(str(response))