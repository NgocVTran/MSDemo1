import os
import sys
# from dotenv import load_dotenv
from math import comb

from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings



# load_dotenv('.env')

import os
os.environ["AZURE_OPENAI_API_KEY"] = "e971867bb3a34fca9edad6ab96c35ca1"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://apac-openai-test.openai.azure.com/"

# model
llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_deployment="apac-gpt-35-turbo",
    temperature=0.3
)

embedding = AzureOpenAIEmbeddings(
# embedding = OpenAIEmbeddings(
    model="apac-text-embedding-3-large",
    openai_api_version="2022-12-01",
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],

)


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

# Convert the document chunks to embedding and save them to the vector store
# vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
# vectordb = Chroma.from_documents(documents,embedding=AzureOpenAIEmbeddings(model="text-embedding-3-large"),
vectordb = Chroma.from_documents(documents,
                                embedding=embedding,
                                persist_directory="./data")
vectordb.persist()



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

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the DocBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = pdf_qa.invoke(
        {"question": query, "chat_history": chat_history})
    print(f"{white}Answer: " + result["answer"])
    chat_history.append((query, result["answer"]))