import streamlit as st
import random
import time

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
import os

os.environ["AZURE_OPENAI_API_KEY"] = "e971867bb3a34fca9edad6ab96c35ca1"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://apac-openai-test.openai.azure.com/"

# model
llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_deployment="apac-gpt-35-turbo",
    temperature=0.5
)

#------------ db setup --------------
from langchain.prompts.chat import ChatPromptTemplate
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         # """
         #  You are a helpful AI assistant expert in querying SQL Database to
         #  find answers to user's question, in Vietnamese.
         # """

         """
         Bạn là một AI assistant expert Việt Nam, chuyên query SQL database để trả lời những câu hỏi của người dùng.
         """
         ),
        ("user", "{question}\n ai: "),
    ]
)

from langchain.agents import AgentType, create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# db = SQLDatabase.from_uri("sqlite:///Chinook.db")
db = SQLDatabase.from_uri("sqlite:///storage.sqlite")

sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_toolkit.get_tools()

sqldb_agent = create_sql_agent(
    llm=llm,
    toolkit=sql_toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)
#------------ end --------------



# Streamed response emulator
def response_generator(input_text):
    ans = sqldb_agent.run(final_prompt.format(
        question=input_text
    ))

    response = str(ans)

    for word in response.split():
        yield word + " "
        time.sleep(0.05)

st.title("Simple chat")

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