import streamlit as st
from ingest import load_and_chunk
from rag_pipeline import get_vectorstore, generate_answer
from arize.pandas.logger import Client as ArizeClient
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

st.title("🧠 AI Clone Chatbot with OpenAI")

url = st.text_input("Enter a webpage URL (e.g., blog or wiki):")

if url:
    chunks = load_and_chunk(url)
    st.success(f"Loaded {len(chunks)} chunks.")
    vectorstore = get_vectorstore(chunks)
    retriever = vectorstore.as_retriever()

    query = st.text_input("Ask a question based on the content:")
    if query:
        answer, docs = generate_answer(query, retriever)
        st.subheader("Answer:")
        st.write(answer)

        st.subheader("🔍 Top Retrieved Context:")
        for doc in docs[:3]:
            st.markdown(f"> {doc.page_content[:300]}...")

        # Optional Arize logging
        try:
            arize = ArizeClient(space_id=os.getenv("ARIZE_SPACE_ID"),
                                api_key=os.getenv("ARIZE_API_KEY"))
            df = pd.DataFrame([{"query": query, "response": answer}])
            arize.log_dataframe(df, model_name="ai-clone-rag", version="v1")
            st.success("Logged to Arize ✅")
        except Exception as e:
            st.warning(f"Arize logging failed: {e}")
