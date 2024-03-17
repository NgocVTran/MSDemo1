import streamlit as st
import random
import time

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
import os


from utils.llm_model import llm
from utils.embedding import embedding


from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings


#------------- document load ------------------
documents = []
# Create a List of Documents from all of our files in the ./docs folder
for file in os.listdir("docs"):
    if file.endswith(".pdf"):
        pdf_path = "./docs/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = "./docs/" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        text_path = "./docs/" + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())


# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
documents = text_splitter.split_documents(documents)


vectordb = Chroma.from_documents(documents,
                                embedding=embedding,
                                persist_directory="./data")
vectordb.persist()


#------------------- prompt ----------------------------
from langchain_core.prompts import PromptTemplate

prompt_template = """
    Mở đầu, bạn chỉ cần hỏi khách thông tin về màu sắc và cỡ áo một cách ngắn gọn. 
    Chỉ hội thoại ngắn gọn trọng tâm câu hỏi một cách lịch sự. 
    Không cung cấp quá nhiều thông tin không liên quan đến câu hỏi.
    Nếu không tìm thấy câu trả lời thì trả lời là không biết, không được cố ý tạo câu trả lời không đúng.
    {context}

    Question: {question}
    Helpful Answer:
    """

final_prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# create our Q&A chain
pdf_qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False,
    combine_docs_chain_kwargs={"prompt": final_prompt}
)

chat_history = [

]
# pdf_qa.invoke({"question": query, "chat_history": chat_history})

# Streamed response emulator
def response_generator(input_text):
    ans = pdf_qa.invoke({"question": input_text, "chat_history": chat_history})

    response = str(ans["answer"])

    for word in response.split():
        yield word + " "
        time.sleep(0.05)

st.title("Assistant Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})