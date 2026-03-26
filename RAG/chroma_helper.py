import os

from chromadb import HttpClient
from chromadb import Settings

from langchain_chroma import Chroma

class ChromaDBHelper(object):

    def __init__(self, host="localhost", port=8000):
        super().__init__()  
        print(f"Connecting to {host}:{port}")
        self.client = HttpClient(host=host, port=port)

    def get_chroma(self, collection_name, embed):
        store = Chroma(client=self.client, collection_name=collection_name, embedding_function=embed)
        return store

    def delete_collection(self, collection_name):
        if len(collection_name) > 0:
            print(f"Deleting {collection_name}")
            self.client.delete_collection(name=collection_name)
