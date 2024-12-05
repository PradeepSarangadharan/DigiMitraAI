import os
from typing import Dict, Optional, List
from dotenv import load_dotenv
from pathlib import Path
import traceback
from agents.google_audio_agent import GoogleAudioAgent
from agents.rag_agent import RAGAgent
from agents.llm_agent import LLMAgent
from agents.audio_agent import AudioAgent
import json
from agents.multilingual_agent import MultilingualAgent

class ManagerAgent:
    def __init__(self, 
             rag_confidence_threshold: float = 0.75,
             audio_confidence_threshold: float = 0.7,
             vector_store_path: str = "data/vector_store"):
        # Load environment variables
        self._load_environment()
    
        
        # Initialize agents
        self.rag_agent = RAGAgent(vector_store_path=vector_store_path)
        self.llm_agent = LLMAgent()
        self.audio_agent = AudioAgent()
        self.multilingual_agent = MultilingualAgent()
        self.google_audio_agent = GoogleAudioAgent()
        
        # Set confidence thresholds
        self.rag_confidence_threshold = rag_confidence_threshold
        self.audio_confidence_threshold = audio_confidence_threshold

            # Update process_multilingual_query method to use google_audio_agent for audio:
        if audio_file:
            audio_result = self.google_audio_agent.process_audio_query(
                audio_file,
                source_language,
                target_language
            )
        
        print("All agents initialized successfully")

    def _load_environment(self):
        """Load environment variables"""
        try:
            current_dir = Path(__file__).parent.parent
            env_path = current_dir / '.env'
            
            if env_path.exists():
                load_dotenv(env_path)
                os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
            else:
                raise ValueError("No .env file found!")
            
            if not os.getenv('OPENAI_API_KEY'):
                raise ValueError("OPENAI_API_KEY not found in environment variables!")
            
            print("Environment variables loaded successfully")
        except Exception as e:
            print(f"Error loading environment variables: {str(e)}")
            raise

    def process_query(self, query: str, audio_file: Optional = None) -> Dict:
        """Process a query using the appropriate agent(s)"""
        try:
            if audio_file:
                audio_result = self._process_audio_input(audio_file)
                if not audio_result["success"]:
                    return audio_result
                query = audio_result["text"]

            # Clear previous memory
            self.rag_agent.clear_memory()
            self.llm_agent.clear_memory()

            # Process with RAG
            print(f"\nProcessing query: {query}")
            try:
                rag_response = self.rag_agent.process_query(query)
                confidence = rag_response["confidence"]
                domain_relevant = rag_response["domain_relevant"]
                has_sources = rag_response["has_sources"]
                semantic_match = rag_response.get("semantic_match", 0.0)
                debug_info = rag_response.get("debug_info", {})
                
                print(f"RAG confidence: {confidence}")
                print(f"Domain relevant: {domain_relevant}")
                print(f"Has sources: {has_sources}")
                print(f"Semantic match: {semantic_match}")
                
                # Case 1: Very high confidence RAG response
                if confidence >= self.rag_confidence_threshold and semantic_match >= 0.85:
                    print("Using RAG response (high confidence)")
                    return {**rag_response, "text": query}
                
                # Case 2: Exact match
                if rag_response.get("exact_match", False):
                    print("Using RAG response (exact match)")
                    return {**rag_response, "text": query}
                
                # Case 3: Domain relevant query
                if domain_relevant:
                    print("Domain relevant query, processing...")
                    
                    # Get LLM response
                    llm_response = self.llm_agent.process_query(query)
                    
                    # Combine responses only if RAG has moderately relevant info
                    if has_sources and semantic_match >= 0.6:
                        print("Combining RAG and LLM responses")
                        combined = self._combine_responses(rag_response, llm_response)
                        return {**combined, "text": query}
                    
                    # Use LLM response for domain-relevant queries without good matches
                    print("Using LLM response for domain-relevant query")
                    return {**llm_response, "text": query}
                
                # Case 4: Not domain relevant
                return {
                    "answer": "I'm an Aadhaar assistance chatbot. Could you please ask a question related to Aadhaar services?",
                    "confidence": 1.0,
                    "source": "System",
                    "text": query
                }

            except Exception as e:
                print(f"RAG processing error: {str(e)}")
                if domain_relevant:
                    return self._process_llm_fallback(query)
                raise

        except Exception as e:
            error_msg = f"Error in process_query: {str(e)}"
            print(error_msg)
            return {
                "answer": "I apologize, but I encountered an error processing your request. Please try again.",
                "source": "Error Handler",
                "confidence": 0.0,
                "text": query if query else "Audio processing failed"
            }

    def _process_audio_input(self, audio_file) -> Dict:
        """Process audio input and handle errors"""
        try:
            audio_result = self.audio_agent.process_audio(audio_file)
            
            if not audio_result["success"]:
                return {
                    "success": False,
                    "answer": f"Audio processing error: {audio_result.get('error', 'Unknown error')}",
                    "source": "Audio Processing",
                    "confidence": 0.0
                }
            
            if audio_result["confidence"] < self.audio_confidence_threshold:
                return {
                    "success": False,
                    "answer": "The audio wasn't clear enough. Could you please repeat or type your question?",
                    "source": "Audio Processing",
                    "confidence": audio_result["confidence"]
                }
            
            return audio_result
            
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return {
                "success": False,
                "answer": "Error processing audio input",
                "source": "Audio Processing",
                "confidence": 0.0
            }

    def _process_llm_fallback(self, query: str) -> Dict:
        """Process query with LLM as fallback"""
        try:
            return self.llm_agent.process_query(query)
        except Exception as e:
            print(f"Error in LLM fallback: {str(e)}")
            return {
                "answer": "I apologize, but I'm having trouble processing your request. Please try again.",
                "source": "Error Handler",
                "confidence": 0.0
            }

    def _combine_responses(self, rag_response: Dict, llm_response: Dict) -> Dict:
        """Combine RAG and LLM responses"""
        combined_answer = f"""Based on our knowledge base and AI analysis:
        
        {rag_response['answer']}
        
        Here's response from LLM: {llm_response['answer']}"""
        
        return {
            "answer": combined_answer,
            "sources": rag_response.get("sources", []) + [llm_response["source"]],
            "confidence": max(rag_response["confidence"], llm_response["confidence"])
        }
    def initialize_knowledge_base(self, documents: Optional[List[str]] = None) -> None:
        """Initialize the RAG knowledge base"""
        try:
            print("Starting knowledge base initialization...")
            if documents:
                print("Warning: Direct document initialization is deprecated. Please use the FAQ converter utility to update the JSON knowledge base.")
                
            # Initialize vector store from JSON
            self.rag_agent.initialize_vector_store()
            print("Knowledge base initialized successfully")
        except Exception as e:
            print(f"Error initializing knowledge base: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def update_knowledge_base(self) -> None:
        """Update the knowledge base from JSON"""
        try:
            print("Updating knowledge base from JSON...")
            # Re-initialize vector store with latest JSON data
            self.rag_agent.initialize_vector_store()
            print("Knowledge base updated successfully")
        except Exception as e:
            print(f"Error updating knowledge base: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def process_multilingual_query(self, 
                                query: Optional[str] = None,
                                audio_file: Optional = None,
                                source_language: str = 'english',
                                target_language: str = 'english') -> Dict:
        """Process query in specified language"""
        try:
            print("\nProcessing Query:")
            print(f"Source Language: {source_language}")
            print(f"Target Language: {target_language}")
            if not hasattr(self, 'multilingual_agent'):
                self.multilingual_agent = MultilingualAgent()

            # Process audio if provided
            if audio_file:
                print(f"\nProcessing audio in {source_language}...")
                audio_result = self.multilingual_agent.process_audio_query(
                    audio_file,
                    source_language
                )
                
                if not audio_result["success"]:
                    return {
                        "answer": f"Error processing audio: {audio_result.get('error', 'Unknown error')}",
                        "success": False
                    }
                
                # Use English translation for processing
                original_query = audio_result["text"]  # Original language text
                query = audio_result["original_text"]  # English translation
                print(f"Original language text: {original_query}")
                print(f"English query: {query}")
            else:
                # Handle text input
                original_query = query
                if source_language != 'english':
                    # Translate to English for processing
                    translation = self.multilingual_agent.translate_text(
                        query, 
                        source_language, 
                        'english'
                    )
                    if not translation["success"]:
                        return translation
                    query = translation["text"]
                    print(f"Translated query to English: {query}")
                else:
                    query = original_query

            # Process query in English using RAG/LLM
            print(f"\nProcessing English query: {query}")
            try:
                # Try RAG first
                rag_response = self.rag_agent.process_query(query)
                
                if rag_response["confidence"] >= self.rag_confidence_threshold:
                    response = rag_response
                else:
                    # Use LLM if RAG confidence is low
                    llm_response = self.llm_agent.process_query(query)
                    response = llm_response
                    
            except Exception as e:
                print(f"Error in query processing: {str(e)}")
                response = self.llm_agent.process_query(query)  # Fallback to LLM

            # Translate response if needed
            if target_language != 'english':
                translation = self.multilingual_agent.translate_text(
                    response["answer"],
                    'english',
                    target_language
                )
                if translation["success"]:
                    response["original_answer"] = response["answer"]
                    response["answer"] = translation["text"]

            return response

        except Exception as e:
            print(f"Error in process_multilingual_query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "answer": "An error occurred processing your query",
                "source_language": source_language,
                "target_language": target_language
            }