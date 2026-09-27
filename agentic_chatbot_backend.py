from langgraph.graph import StateGraph ,START,END
from typing import Annotated ,TypedDict
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(model="openai/gpt-oss-20b",api_key=groq_api_key)

from langgraph.graph.message import add_messages
class ChatState(TypedDict):
  messages:Annotated[list[BaseMessage],add_messages]
def chat_node(state:ChatState):
  messages=state['messages']
  response=llm.invoke(messages)
  return {
      'messages':[response]
  }
  
checkpoint=MemorySaver()
graph=StateGraph(ChatState)
graph.add_node('chat_node',chat_node)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)
chatbot=graph.compile(checkpointer=checkpoint)