from llama_index import PromptTemplate
from tqdm.asyncio import tqdm
from llama_index import QueryBundle
from llama_index.retrievers import BaseRetriever
from typing import List
from llama_index.schema import NodeWithScore
# import nest_asyncio

# nest_asyncio.apply()

## QUERY GENERATION / REWRITING QUERY

def generate_queries(llm, query_gen_prompt, query_str: str, num_queries: int = 4):
    # Generate query generate prompt with llama index api
    query_gen_prompt = PromptTemplate(query_gen_prompt)
    # Insert format to generate prompt
    fmt_prompt = query_gen_prompt.format(
        num_queries=num_queries - 1, query=query_str
    )
    # feed prompt into llm
    response = llm.complete(fmt_prompt)
    # split response into queries
    queries = response.text.split("\n")

    return queries


## PERFORM VECTOR SEARCH FOR EACH QUERY

async def run_queries(queries, retrievers):
    """Run queries against retrievers."""
    tasks = []
    for query in queries:
        for i, retriever in enumerate(retrievers):
            tasks.append(retriever.aretrieve(query))

    task_results = await tqdm.gather(*tasks)

    results_dict = {}
    for i, (query, query_result) in enumerate(zip(queries, task_results)):
        results_dict[(query, i)] = query_result

    return results_dict


# results_dict = await run_queries(queries, [vector_retriever, bm25_retriever])


## PERFORM RANK FUSION
def fuse_results(results_dict, similarity_top_k: int = 2):
    """Fuse results."""
    k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
    fused_scores = {}
    text_to_node = {}

    # compute reciprocal rank scores
    for nodes_with_scores in results_dict.values():
        for rank, node_with_score in enumerate(
            sorted(
                nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True
            )
        ):
            text = node_with_score.node.get_content()
            text_to_node[text] = node_with_score
            if text not in fused_scores:
                fused_scores[text] = 0.0
            fused_scores[text] += 1.0 / (rank + k)

    # sort results
    reranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    # adjust node scores
    reranked_nodes: List[NodeWithScore] = []
    for text, score in reranked_results.items():
        reranked_nodes.append(text_to_node[text])
        reranked_nodes[-1].score = score

    return reranked_nodes[:similarity_top_k]

# final_results = fuse_results(results_dict)


class FusionRetriever(BaseRetriever):
    """Ensemble retriever with fusion."""

    def __init__(
        self,
        llm,
        retrievers: List[BaseRetriever],
        similarity_top_k: int = 2,
    ) -> None:
        
        """Init params."""
        self._retrievers = retrievers
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        queries = generate_queries(llm, query_str, num_queries=4)
        results_dict = run_queries(queries, [vector_retriever, bm25_retriever])
        final_results = fuse_results(
            results_dict, similarity_top_k=self._similarity_top_k
        )

        return final_results