import os
import streamlit as st
import pickle
import langchain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS  
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

from model import llm

loaders = UnstructuredURLLoader(urls=[
    "https://www.moneycontrol.com/news/business/markets/wall-street-rises-as-tesla-soars-on-ai-optimism-11351111.html",
    "https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html"
])
data = loaders.load() 
len(data)
# text="""
# Governor RN Ravi has repeatedly" clashed with the MK Stalin government in Tamil NaduNew Delhi:
# In a massive win for the MK Stalin-led Tamil Nadu government, the Supreme Court today ruled that Governor RN Ravi's decision to withhold assent to 10 key Bills was "illegal" and "arbitrary". The court ruled that the Governor cannot reserve Bills for the President after withholding assent.
# "The action of the Governor to reserve the 10 bills for the President is illegal and arbitrary. Thus, the action is set aside. All actions taken by the Governor thereto for the 10 bills are set aside. These Bills shall be deemed to be cleared from the date it was re-presented to the Governor," the bench of Justice JB Pardiwala and Justice R Mahadevan said. The court said Governor Ravi did not act in "good faith".
# Tamil Nadu Chief Minister and DMK chief MK Stalin described the verdict as "historic". "It's a big victory not just for Tamil Nadu but for all Indian states. DMK will continue to struggle for and win state autonomy and federal polity," he said. """

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# As data is of type documents we can directly use split_documents over split_text in order to get the chunks.
docs = text_splitter.split_documents(data)
print(len(docs))

embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

file_path = "vector_store.pkl"
with open(file_path, "wb") as f:
    pickle.dump(vector_store, f)

if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorIndex = pickle.load(f)

prompt_template = """Use the following context to answer the question at the end. 
If you don't know the answer, just say you don't know — don't try to make up an answer.

{context}

Question: {question}
Answer:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create the LLM chain using your existing `llm` (like GoogleGenerativeAI or any other)
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Combine context into a final answer
combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

# Final retrieval chain
chain = RetrievalQAWithSourcesChain(
    retriever=vectorIndex.as_retriever(),
    combine_documents_chain=combine_documents_chain
)

query = "what is the price of Tiago iCNG?"
# query = "what are the main features of punch iCNG?"

langchain.debug=True

chain({"question": query}, return_only_outputs=True)
