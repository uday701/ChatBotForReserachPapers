from pathlib import Path
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class RagImplementation:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.metadatas = {}

        output_parent_folder = Path(output_folder)

        for json_file_path in output_parent_folder.rglob("*.json"):
            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)

                if isinstance(data, list) and len(data) > 0:
                    # Get the first object from the JSON file
                    first_object = data[0]
                    new_object = {
                        'title': first_object['title'],
                        'authors': first_object['authors'],
                        'filename': first_object['filename']
                    }

                    # Use the new_object as the key and json_file_path as the value in the metadatas dictionary
                    self.metadatas[json.dumps(new_object)] = str(json_file_path)

        self.docs = [
            Document(page_content=key, metadata={"index": index})
            for index, key in enumerate(self.metadatas.keys())
        ]
        vector_store = FAISS.from_documents(self.docs, self.embedding_function)
        retriever = vector_store.as_retriever()
        llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1", model_kwargs={"temperature": 1, "max_length": 6000, "max_new_tokens": 4096})

        self.multi_query_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

    def process(self, query):

        relevant_documents = self.multi_query_retriever.get_relevant_documents(query)
        filepaths = []
        metapaths = []
        documents = []

        for document in relevant_documents:
            if document.page_content in self.metadatas:
                # Retrieve the corresponding value from metadatas
                file_path = self.metadatas[document.page_content]
                metapaths.append(file_path)
                new_filename = file_path.rsplit('.pdf', 1)[0] + '.pdf.tei_text.txt'
                filepaths.append(new_filename)
                

        for i in range(len(filepaths)):
            try:
                with open(filepaths[i], 'r',encoding="utf-8") as text_file, open(metapaths[i], 'r',encoding="utf-8") as meta_file:
                    data = json.load(meta_file)
                    
                    for line1, line2 in zip(text_file, data):
                        
                        document = Document(page_content=line1, metadata={"index": json.dumps(line2)})
                        documents.append(document)
            except FileNotFoundError as e:
                print(f"File not found: {e.filename}")
            except Exception as e:
                print(f"An error occurred: {e}")

        vector_store1 = FAISS.from_documents(documents, self.embedding_function)
        retriever1 = vector_store1.as_retriever()
        return retriever1
