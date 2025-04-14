import os
import streamlit as st
import time
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


st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
file_path = "vector_store.pkl"
if process_url_clicked:
    loaders = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loaders.load() 
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    vector_store = FAISS.from_documents(
        documents=docs,
        embedding=embedding_model
    )
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)
    

    with open(file_path, "wb") as f:
        pickle.dump(vector_store, f)
        
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorIndex = pickle.load(f)
            # prompt_template = """Use the following context to answer the question at the end. 
            # If you don't know the answer, just say you don't know — don't try to make up an answer.

            # {context}

            # Question: {question}
            # Answer:"""

            # prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

            # # Create the LLM chain using your existing `llm` (like GoogleGenerativeAI or any other)
            # llm_chain = LLMChain(llm=llm, prompt=prompt)

            # # Combine context into a final answer
            # combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

            # # Final retrieval chain
            # chain = RetrievalQAWithSourcesChain(
            #     retriever=vectorIndex.as_retriever(),
            #     combine_documents_chain=combine_documents_chain
            # )
            # result = chain({"question": query}, return_only_outputs=True) 
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                retriever=vectorIndex.as_retriever(),
                chain_type="stuff"  # internally uses StuffDocumentsChain
            )

            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer") 
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            print("sources",sources)
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
