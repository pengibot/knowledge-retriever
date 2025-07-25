# app.py
import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# App title
st.set_page_config(page_title="🤖 Document Q&A Bot")
st.title("📄 Document Q&A Bot")

# Use the token from Streamlit secrets
try:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except:
    st.error("Hugging Face API token not found in .streamlit/secrets.toml")
    st.stop()

# --- RAG Setup ---
# Load resources directly
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 2})

repo_id = "google/flan-t5-large"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text2text-generation",  # Ensure the task is correctly specified
    temperature=0.2,
    max_new_tokens=512
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# --- Streamlit UI ---
query = st.text_input("Ask a question about your document:")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"query": query})

        st.write("### Answer")
        st.write(result["result"])

        with st.expander("Sources"):
            for source in result["source_documents"]:
                st.write(f"- **Page {source.metadata.get('page', 'N/A')}:** {source.page_content[:250]}...")