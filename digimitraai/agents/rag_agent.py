from typing import Dict, List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import os
from pathlib import Path
import fitz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.domain_checker import DomainChecker

class RAGAgent:
    def __init__(self, model_name: str = "gpt-3.5-turbo", vector_store_path: str = "data/vector_store"):
        self.model_name = model_name
        self.vector_store_path = vector_store_path
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7)
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        self.domain_checker = DomainChecker()
        
        self.qa_template = """You are an expert Aadhaar customer service assistant. You can ONLY answer questions related to Aadhaar and its services.
        If the question is not related to Aadhaar, politely inform that you can only assist with Aadhaar-related queries.
        Use the following context to answer the question at the end.
        If you don't know the answer based on the context, just say "I don't have enough information to answer that question accurately."
        
        Context: {context}

        Question: {question}

        Answer: """

        self.qa_prompt = PromptTemplate(
            template=self.qa_template,
            input_variables=["context", "question"]
        )
        
        self.vector_store = None
        self.qa_chain = None
        
        self._load_vector_store()

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Process PDF document and return chunks with metadata"""
        doc = fitz.open(pdf_path)
        chunks = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            
            # Split text into chunks
            texts = self.text_splitter.split_text(text)
            
            # Create chunks with metadata
            for chunk in texts:
                chunks.append({
                    "content": chunk,
                    "metadata": {
                        "source": pdf_path,
                        "page": page_num + 1
                    }
                })
        
        return chunks

    def _load_vector_store(self):
        """Load or create vector store from JSON FAQs"""
        try:
            # First try to load existing vector store
            vector_store_path = Path(self.vector_store_path)
            if vector_store_path.exists():
                self.vector_store = FAISS.load_local(
                    str(vector_store_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self._initialize_qa_chain()
                print("Successfully loaded existing vector store")
            else:
                # Initialize from JSON if vector store doesn't exist
                self.initialize_vector_store()
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            self.vector_store = None

    def _initialize_qa_chain(self):
        """Initialize the QA chain with the vector store"""
        try:
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": self.qa_prompt}
            )
            print("Successfully initialized QA chain")
        except Exception as e:
            print(f"Error initializing QA chain: {str(e)}")
            self.qa_chain = None

    def initialize_vector_store(self, pdf_paths: List[str]) -> None:
        """Initialize FAISS vector store from PDF documents"""
        all_chunks = []
        for pdf_path in pdf_paths:
            chunks = self.process_pdf(pdf_path)
            all_chunks.extend(chunks)
        
        texts = [chunk["content"] for chunk in all_chunks]
        metadatas = [chunk["metadata"] for chunk in all_chunks]
        
        self.vector_store = FAISS.from_texts(
            texts, 
            self.embeddings,
            metadatas=metadatas
        )
        
        os.makedirs(self.vector_store_path, exist_ok=True)
        self.vector_store.save_local(self.vector_store_path)
        
        self._initialize_qa_chain()

    def update_vector_store(self, new_pdf_paths: List[str]) -> None:
        """Add new PDF documents to existing vector store"""
        new_chunks = []
        for pdf_path in new_pdf_paths:
            chunks = self.process_pdf(pdf_path)
            new_chunks.extend(chunks)
        
        texts = [chunk["content"] for chunk in new_chunks]
        metadatas = [chunk["metadata"] for chunk in new_chunks]
        
        if self.vector_store is None:
            self._load_vector_store()
        
        if self.vector_store:
            self.vector_store.add_texts(texts, metadatas=metadatas)
            self.vector_store.save_local(self.vector_store_path)
        else:
            self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
            self.vector_store.save_local(self.vector_store_path)
        
        self._initialize_qa_chain()    

    def _load_vector_store(self):
        try:
            if os.path.exists(self.vector_store_path):
                self.vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self._initialize_qa_chain()
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            self.vector_store = None    

    def _is_domain_relevant(self, query: str) -> bool:
        """Check if query is relevant to Aadhaar domain"""
        domain_keywords = [
            'aadhaar', 'aadhar', 'uid', 'uidai', 'biometric', 'enrollment', 'enrolment',
            'demographic', 'authentication', 'ekyc', 'kyc', 'resident', 'virtual id',
            'update', 'correction', 'verification', 'identity', 'card', 'number',
            'unique identification', 'address', 'mobile', 'email', 'fingerprint', 'iris',
            'face', 'photo', 'otp', 'masked', 'mandatory', 'optional', 'register'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in domain_keywords)

    def _calculate_confidence(self, query: str, source_documents) -> Dict:
        try:
            response = {
                "confidence": 0.0,
                "domain_relevant": False,
                "has_sources": bool(source_documents),
                "exact_match": False,
                "semantic_match": 0.0,
                "debug_info": {}
            }
            
            response["domain_relevant"] = self._is_domain_relevant(query)
            
            if not response["domain_relevant"]:
                return response
            
            exact_match = self._find_exact_faq_match(query)
            if exact_match['confidence'] > 0.0:
                response.update({
                    "confidence": exact_match['confidence'],
                    "exact_match": True,
                    "semantic_match": 1.0
                })
                return response
            
            if not source_documents:
                response.update({
                    "confidence": 0.2,
                    "semantic_match": 0.0,
                    "debug_info": {"reason": "no_sources"}
                })
                return response
                
            query_embedding = self.embeddings.embed_query(query)
            similarities = []
            matched_texts = []
            
            for doc in source_documents:
                doc_text = doc.page_content
                if "Q:" in doc_text and "A:" in doc_text:
                    question_part = doc_text.split("A:")[0].replace("Q:", "").strip()
                else:
                    question_part = doc_text

                # Check for key terms matching
                query_terms = set(query.lower().split())
                doc_terms = set(question_part.lower().split())
                term_overlap = len(query_terms.intersection(doc_terms)) / len(query_terms)

                doc_embedding = self.embeddings.embed_query(question_part)
                similarity = float(cosine_similarity(
                    np.array(query_embedding).reshape(1, -1),
                    np.array(doc_embedding).reshape(1, -1)
                )[0][0])
                
                # Adjust similarity based on term overlap
                adjusted_similarity = similarity * (0.7 + 0.3 * term_overlap)
                
                similarities.append(adjusted_similarity)
                matched_texts.append({
                    "text": question_part[:100] + "...",
                    "similarity": adjusted_similarity,
                    "term_overlap": term_overlap
                })

            max_similarity = max(similarities) if similarities else 0.0
            response["semantic_match"] = max_similarity
            
            # More stringent confidence calculation
            if max_similarity >= 0.95:
                confidence = 0.95
            elif max_similarity >= 0.90:
                confidence = 0.85
            elif max_similarity >= 0.80:
                confidence = 0.5  # Significantly reduced for partial matches
            else:
                confidence = 0.2  # Force low confidence for poor matches
            
            # Check for mandatory/optional specific terms
            if 'mandatory' in query.lower() or 'compulsory' in query.lower():
                matched_mandatory = any('mandatory' in doc.page_content.lower() or 
                                    'compulsory' in doc.page_content.lower() 
                                    for doc in source_documents)
                if not matched_mandatory:
                    confidence *= 0.5  # Reduce confidence if specific terms not found
            
            relevant_sources = len([s for s in similarities if s > 0.8])  # Increased threshold
            if relevant_sources == 0:
                confidence *= 0.5
            
            response.update({
                "confidence": confidence,
                "debug_info": {
                    "max_similarity": max_similarity,
                    "relevant_sources": relevant_sources,
                    "final_confidence": confidence,
                    "matched_texts": matched_texts
                }
            })
            
            return response

        except Exception as e:
            print(f"Error in confidence calculation: {str(e)}")
            return {
                "confidence": 0.0,
                "domain_relevant": False,
                "has_sources": False,
                "exact_match": False,
                "semantic_match": 0.0,
                "debug_info": {"error": str(e)}
            }

    def process_query(self, query: str) -> Dict:
        try:
            is_relevant, relevance_score = self.domain_checker.is_domain_relevant(query)
            
            if not is_relevant:
                return {
                    "answer": "I apologize, but I can only assist with questions related to Aadhaar and its services. Your question appears to be about something else. Please feel free to ask any Aadhaar-related questions.",
                    "confidence": 1.0,
                    "sources": []
                }

            if not self.qa_chain:
                self._load_vector_store()
                if not self.qa_chain:
                    raise ValueError("QA chain not initialized.")
            
            print(f"\nRAG Processing:")
            print(f"Query: {query}")


            # Get response from QA chain
            result = self.qa_chain({
                "question": query,
                "chat_history": []
            })
        
            source_docs = result.get("source_documents", [])
            confidence_info = self._calculate_confidence(query, source_docs)
            
            # Format sources
            sources = []
            for doc in source_docs:
                if hasattr(doc, 'metadata'):
                    sources.append(f"Source: {doc.metadata.get('source', 'Unknown')}, "
                                f"Page: {doc.metadata.get('page', 'Unknown')}")
            
            print(f"Confidence Score: {confidence_info}")
            print("Relevant Chunks:")
            for doc in source_docs:
                print(f"- Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}")
                print(f"  Content: {doc.page_content[:200]}...")

            return {
                "answer": result["answer"],
                "sources": [f"Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}" for doc in source_docs],
                "confidence": confidence_info["confidence"],
                "domain_relevant": confidence_info["domain_relevant"],
                "has_sources": confidence_info["has_sources"],
                "exact_match": confidence_info["exact_match"],
                "semantic_match": confidence_info["semantic_match"],
                "debug_info": confidence_info.get("debug_info", {})
            }
         
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            raise
    
    def _find_exact_faq_match(self, query: str) -> Dict:
        """Find exact match in vector store"""
        try:
            if not self.vector_store:
                return {"confidence": 0.0, "match": None}

            results = self.vector_store.similarity_search_with_score(query, k=1)
            if results:
                doc, score = results[0]
                exact_match_threshold = 0.95
                if score >= exact_match_threshold:
                    return {"confidence": 1.0, "match": doc}
                elif score >= 0.8:
                    return {"confidence": 0.9, "match": doc}
            
            return {"confidence": 0.0, "match": None}
            
        except Exception as e:
            print(f"Error in exact match: {str(e)}")
            return {"confidence": 0.0, "match": None}

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()