from typing import Optional, Sequence
from llama_index.schema import Document
import requests
from pathlib import Path
from llama_index import download_loader
from llama_index.node_parser import SimpleNodeParser
from llama_index import VectorStoreIndex, ServiceContext

def read_pdf_from_link(pdf_path:str, local_path:str=None)-> Optional[Sequence[Document]]:
    # Init pdf reading tool
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()
    # if link given is web-base
    if pdf_path.startswith("https"):
        r = requests.get(pdf_path)
        # download and write to localfile
        with open(local_path, 'wb') as f:
            f.write(r.content)
        
        # Load to corpus text
        doc = loader.load_data(file=Path(local_path))[0]
    # if link given is local
    # Load to corpus text
    else: doc = loader.load_data(file=Path(pdf_path))[0]
    
    # # parse corpus to nodes
    # nodes = SimpleNodeParser().get_nodes_from_documents([doc])

    # return nodes
    return doc

def get_vector_store_index(documents:Document)-> None:
    service_context = ServiceContext.from_defaults(chunk_size=1024)
    index = VectorStoreIndex.from_documents(documents, service_context= service_context)
    return index