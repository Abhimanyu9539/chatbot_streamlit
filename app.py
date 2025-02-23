import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.base import BaseCallbackHandler

# Streamlit Page Configuration
st.set_page_config(page_title="Gemini AI Chatbot", page_icon="🤖")
st.title("🤖 Gemini AI Chatbot")

# Define StreamHandler for real-time streaming response updates
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Initialize Gemini AI Chat Model (Ensure API key is set in Streamlit Secrets)
gemini = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1, streaming=True)

# Define System Prompt
SYS_PROMPT = "You are a helpful AI assistant. Answer questions to the best of your ability."

# Create a Chat Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYS_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Define a conversation chain
llm_chain = prompt | gemini

# Store chat history in Streamlit session state
streamlit_msg_history = StreamlitChatMessageHistory()

conversation_chain = RunnableWithMessageHistory(
    llm_chain,
    lambda session_id: streamlit_msg_history,  # Access memory
    input_messages_key="input",
    history_messages_key="history",
)

# Show welcome message
if len(streamlit_msg_history.messages) == 0:
    streamlit_msg_history.add_ai_message("Hello! How can I assist you today?")

# Display past chat messages
for msg in streamlit_msg_history.messages:
    st.chat_message(msg.type).write(msg.content)

# Handle new user inputs
if user_prompt := st.chat_input():
    st.chat_message("human").write(user_prompt)

    with st.chat_message("ai"):
        stream_handler = StreamHandler(st.empty())
        config = {"configurable": {"session_id": "any"}, "callbacks": [stream_handler]}
        response = conversation_chain.invoke({"input": user_prompt}, config)
