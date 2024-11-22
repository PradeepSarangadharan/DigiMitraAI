# agents/llm_agent.py

from typing import Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

class LLMAgent:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=model_name)
        self.memory = ConversationBufferMemory(return_messages=True)
        
        self.prompt = ChatPromptTemplate.from_template("""
            You are an expert Aadhaar customer service assistant. Your role is to provide accurate
            and helpful information about Aadhaar services, processes, and requirements.
            
            Question: {question}
            
            Assistant: Let me help you with that query about Aadhaar.
        """)
        
        self.chain = self.prompt | self.llm

    def process_query(self, query: str) -> Dict:
        """Process a query using the LLM"""
        try:
            response = self.chain.invoke({"question": query})
            
            return {
                "answer": response.content,
                "source": "LLM-generated response",
                "confidence": 0.8
            }
        except Exception as e:
            print(f"Error in LLM processing: {str(e)}")
            raise

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()