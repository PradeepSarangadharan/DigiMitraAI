# agents/llm_agent.py

from typing import Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from utils.domain_checker import DomainChecker

class LLMAgent:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=model_name)
        self.memory = ConversationBufferMemory(return_messages=True)
        
        self.domain_checker = DomainChecker()
        
        self.prompt = ChatPromptTemplate.from_template("""
            You are an expert Aadhaar customer service assistant. You can ONLY answer questions related to Aadhaar and its services.
            If the question is not related to Aadhaar, politely inform that you can only assist with Aadhaar-related queries.
            
            Question: {question}
            
            Assistant: Let me help you with your query about Aadhaar.
            First, let me check if this is related to Aadhaar services.
        """)
        
        self.chain = self.prompt | self.llm

    def process_query(self, query: str) -> Dict:
        try:
            # Check domain relevance
            is_relevant, relevance_score = self.domain_checker.is_domain_relevant(query)
            
            if not is_relevant:
                return {
                    "answer": "I apologize, but I can only assist with questions related to Aadhaar and its services. Your question appears to be about something else. Please feel free to ask any Aadhaar-related questions.",
                    "confidence": 1.0,
                    "source": "Domain Check"
                }
            
            # Process relevant query
            response = self.chain.invoke({"question": query})
            
            return {
                "answer": response.content,
                "source": "LLM-generated response",
                "confidence": 0.8 * relevance_score  # Adjust confidence based on relevance
            }
        except Exception as e:
            print(f"Error in LLM processing: {str(e)}")
            raise

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()