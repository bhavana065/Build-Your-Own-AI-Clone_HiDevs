from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
embedder = OpenAIEmbeddings()

from langchain.vectorstores import Chroma

def get_vectorstore(chunks):
    return Chroma.from_documents(chunks, embedding=embedder)

def generate_answer(query, retriever):
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Context:\n{context}\n\n### Question:\n{query}\n### Answer:"

    # Use LangChain chat model for completion
    response = llm.chat([{"role": "user", "content": prompt}])
    return response.content, docs
