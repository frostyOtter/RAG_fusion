# Retriever Augmented Generation Fusion (RAG-Fusion) (WIP)
## Overview: The code accompanying this README is an implementation of RAG-Fusion, a search methodology that aims to bridge the gap between traditional search paradigms and the multifaceted dimensions of human queries, readmore in medium: Forget RAG, the future is RAG-fusion

## What The Code Does:
    1. Query Generation: The system starts by generating multiple queries from a user's initial query using OpenAI's GPT model.

    2. Vector Search: Conducts vector-based searches on each of the generated queries to retrieve relevant documents from a predefined set.

    3. Reciprocal Rank Fusion: Applies the Reciprocal Rank Fusion algorithm to re-rank the documents based on their relevance across multiple queries.

    4. Output Generation: Produces a final output consisting of the re-ranked list of documents.

## How to Run the Code:
    - Still work in progress

### TODO:
    - Add LLM init
    - Local embedding model
    - Etc