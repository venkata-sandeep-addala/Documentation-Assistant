import os
from typing import List, Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.messages import ToolMessage

from logger import *


load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = init_chat_model(model="llama-3.1-8b-instant", model_provider='groq',temperature=0.2)

vectorstore = Chroma(persist_directory="chroma_db",
                     collection_name="documentation_assistant", 
                     embedding_function=embedding_model)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str, top_k: int = 4):
    """Retrieve relevant context from the vector store based on the query."""
    try:
        results = vectorstore.similarity_search(query, k=top_k)
        log_info(f"Retrieved {len(results)} relevant documents for the query.")
        serialized_results = '\n\n'.join([f"Source: {result.metadata.get('source', 'Unknown')}\n\nContent: {result.page_content}" for result in results])
        return serialized_results, results
    
    except Exception as e:
        log_error(f"Error retrieving context: {e}")
        return []


def main(query: str):
    log_header("AGENT INITIALIZATION")
    
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a retrieve_context tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        """IMPORTANT:
        - Always include the source URLs used to generate the answer.
        - If multiple sources are used, list all of them.
        - Format sources clearly at the end under "Sources".
        - Do not make up URLs. Only use URLs present in the context."""
        "If you cannot find the answer in the retrieved documentation, say so."
    )
    
    agent = create_agent(model=llm, tools=[retrieve_context], system_prompt=system_prompt)
    log_success("Agent initialized successfully with LLM and tools.")
    
    message = {'role': 'user', 'content': query}

    # Example query to test the agent
    log_info(f"Sending query to agent: {query}")
    response = agent.invoke({'messages': [message]})
    log_info(f"Agent response: {response}")
    
    answer = response['messages'][-1].content
    log_success(f"Final Answer: {answer}")
    
    context_docs =[]
    
    for message in response['messages']:
        if isinstance(message, ToolMessage) and hasattr(message, 'artifact'):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    
    return {'answer': answer, 'context_docs': context_docs}


if __name__ == "__main__":
    result = main("What are the key features of the langchain?")
    print("\nFinal Answer:\n", result['answer'])