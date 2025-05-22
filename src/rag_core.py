"""
Complete RAG system with TF-IDF and Embedding retrievers.
Minimal but production-ready implementation for learning and experimentation.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import time
import numpy as np

# Core ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# LangChain imports  
from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate


class RetrieverBase(ABC):
    """Base class for all retrievers - defines the interface"""
    
    @abstractmethod
    def build_index(self, documents: List[Document]) -> None:
        """Build search index from documents"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """Retrieve documents with similarity scores"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Method name for identification"""
        pass


class TFIDFRetriever(RetrieverBase):
    """
    Traditional keyword-based retrieval using TF-IDF.
    Fast, lightweight, good baseline for comparison.
    """
    
    def __init__(self, max_features: int = 1000, similarity_threshold: float = 0.1):
        """Initialize TF-IDF retriever with basic settings"""
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Words + word pairs
            stop_words='english'
        )
        self.threshold = similarity_threshold
        self.doc_vectors = None
        self.documents = None
    
    def build_index(self, documents: List[Document]) -> None:
        """Convert documents to TF-IDF vectors"""
        self.documents = documents
        doc_texts = [doc.page_content for doc in documents]
        self.doc_vectors = self.vectorizer.fit_transform(doc_texts)
        print(f"📊 TF-IDF index: {len(documents)} docs, {len(self.vectorizer.vocabulary_)} features")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """Find most similar documents using cosine similarity"""
        if not self.documents:
            return []
        
        # Convert query to same vector space as documents
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        
        # Get top results above threshold
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []
        
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= self.threshold:
                results.append((self.documents[idx], score))
        
        return results
    
    @property
    def name(self) -> str:
        return "TF-IDF"


class EmbeddingRetriever(RetrieverBase):
    """
    Modern semantic retrieval using sentence transformers.
    Understands meaning, not just keywords.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.1):
        """Initialize with a sentence transformer model"""
        print(f"🤗 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.threshold = similarity_threshold
        self.doc_embeddings = None
        self.documents = None
    
    def build_index(self, documents: List[Document]) -> None:
        """Convert documents to semantic embeddings"""
        self.documents = documents
        doc_texts = [doc.page_content for doc in documents]
        
        # Create embeddings - this captures semantic meaning
        self.doc_embeddings = self.model.encode(doc_texts)
        print(f"🤗 Embedding index: {len(documents)} docs, {self.doc_embeddings.shape[1]}D vectors")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """Find semantically similar documents"""
        if not self.documents:
            return []
        
        # Encode query into same semantic space
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        
        # Get top results above threshold
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []
        
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= self.threshold:
                results.append((self.documents[idx], score))
        
        return results
    
    @property 
    def name(self) -> str:
        return f"Embeddings ({self.model_name})"


class LLMInterface:
    """Simple interface for LLM text generation"""
    
    def __init__(self, model: str = "llama3.2", temperature: float = 0.3):
        """Initialize with Ollama model"""
        self.llm = Ollama(model=model, temperature=temperature)
        self.model_name = model
        
        # Template for RAG responses
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Answer the question based on the provided context. Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""
        )
    
    def generate_answer(self, context: str, question: str) -> str:
        """Generate answer given context and question"""
        try:
            prompt_text = self.prompt.format(context=context, question=question)
            response = self.llm.invoke(prompt_text)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error generating answer: {e}"


class RAGSystem:
    """
    Main RAG system that orchestrates retrieval and generation.
    Combines any retriever with any LLM.
    """
    
    def __init__(self, retriever: RetrieverBase, llm: LLMInterface):
        """Initialize with retriever and LLM components"""
        self.retriever = retriever
        self.llm = llm
        self.documents = []
    
    def load_documents(self, documents: List[Document]) -> None:
        """Load documents and build search index"""
        self.documents = documents
        self.retriever.build_index(documents)
        print(f"✅ Loaded {len(documents)} documents into {self.retriever.name} system")
    
    def ask(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Ask a question and get answer with detailed metrics.
        This is the main RAG pipeline: Retrieve -> Generate -> Return
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question, top_k)
        
        if not retrieved_docs:
            return {
                "question": question,
                "answer": "No relevant information found.",
                "method": self.retriever.name,
                "model": self.llm.model_name,
                "retrieval_time": time.time() - start_time,
                "docs_found": 0,
                "similarity_scores": [],
                "retrieved_qa_ids": [],
                "context_length": 0,
                "avg_similarity": 0.0
            }
        
        # Step 2: Prepare context from retrieved documents
        context_parts = []
        scores = []
        qa_ids = []
        
        # Debug: Print what we got from retrieval
        print(f"DEBUG: Retrieved {len(retrieved_docs)} documents")
        
        for i, (doc, score) in enumerate(retrieved_docs):
            context_parts.append(doc.page_content)
            scores.append(score)
            
            # Extract QA ID from document metadata with debugging
            qa_id = doc.metadata.get('qa_id', f'unknown_doc_{i}')
            qa_ids.append(qa_id)
            
            # Debug: Print document metadata
            print(f"DEBUG Document {i+1}:")
            print(f"  - QA ID: {qa_id}")
            print(f"  - Score: {score:.3f}")
            print(f"  - Metadata keys: {list(doc.metadata.keys())}")
            print(f"  - Content preview: {doc.page_content[:100]}...")
        
        context = "\n\n".join(context_parts)
        
        # Step 3: Generate answer using LLM
        answer = self.llm.generate_answer(context, question)
        
        # Debug: Print final result
        print(f"DEBUG Final result:")
        print(f"  - QA IDs being returned: {qa_ids}")
        print(f"  - Scores being returned: {scores}")
        print(f"  - Context length: {len(context)}")
        
        # Step 4: Return detailed results
        result = {
            "question": question,
            "answer": answer,
            "method": self.retriever.name,
            "model": self.llm.model_name,
            "retrieval_time": time.time() - start_time,
            "docs_found": len(retrieved_docs),
            "similarity_scores": scores,
            "retrieved_qa_ids": qa_ids,  # This should now contain the actual QA IDs
            "avg_similarity": np.mean(scores) if scores else 0.0,
            "context_length": len(context)
        }
        
        # Debug: Print the keys in the final result
        print(f"DEBUG Result keys: {list(result.keys())}")
        print(f"DEBUG retrieved_qa_ids in result: {result['retrieved_qa_ids']}")
        
        return result
    
    def batch_ask(self, questions: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """Ask multiple questions and return all results"""
        return [self.ask(q, top_k) for q in questions]


class RAGComparison:
    """
    Utility class for comparing different RAG configurations.
    Perfect for experiments and Medium article content.
    """
    
    def __init__(self, documents: List[Document]):
        """Initialize with document collection"""
        self.documents = documents
        
        # Create different RAG systems for comparison
        self.systems = {
            "tfidf": RAGSystem(TFIDFRetriever(), LLMInterface()),
            "embeddings": RAGSystem(EmbeddingRetriever(), LLMInterface())
        }
        
        # Load documents into all systems
        for name, system in self.systems.items():
            system.load_documents(documents)
    
    def compare_question(self, question: str, top_k: int = 3) -> Dict[str, Dict]:
        """Compare how different methods answer the same question"""
        results = {}
        
        for method_name, system in self.systems.items():
            result = system.ask(question, top_k)
            results[method_name] = result
        
        return results
    
    def run_experiment(self, test_questions: List[str]) -> Dict[str, Any]:
        """
        Run full comparison experiment across all test questions.
        Returns comprehensive results perfect for analysis.
        """
        print(f"🧪 Running comparison experiment with {len(test_questions)} questions...")
        
        all_results = {}
        method_stats = {}
        
        # Test each method on all questions
        for method_name, system in self.systems.items():
            print(f"\n🔧 Testing {method_name}...")
            
            method_results = []
            times = []
            similarities = []
            success_count = 0
            
            for question in test_questions:
                result = system.ask(question)
                method_results.append(result)
                
                times.append(result['retrieval_time'])
                if result['docs_found'] > 0:
                    success_count += 1
                    similarities.append(result['avg_similarity'])
            
            # Calculate aggregate statistics
            method_stats[method_name] = {
                "avg_response_time": np.mean(times),
                "success_rate": success_count / len(test_questions),
                "avg_similarity": np.mean(similarities) if similarities else 0.0,
                "total_questions": len(test_questions)
            }
            
            all_results[method_name] = method_results
        
        # Combine results
        experiment_results = {
            "summary": method_stats,
            "detailed_results": all_results,
            "test_questions": test_questions,
            "document_count": len(self.documents)
        }
        
        # Print comparison summary
        print(f"\n📊 EXPERIMENT RESULTS:")
        print(f"{'Method':<12} {'Success Rate':<12} {'Avg Time':<10} {'Avg Similarity':<15}")
        print("-" * 55)
        
        for method, stats in method_stats.items():
            print(f"{method:<12} {stats['success_rate']:<12.2%} "
                  f"{stats['avg_response_time']:<10.2f}s {stats['avg_similarity']:<15.3f}")
        
        return experiment_results


# Factory functions for easy setup
def create_tfidf_rag(documents: List[Document] = None) -> RAGSystem:
    """Create a TF-IDF based RAG system"""
    system = RAGSystem(TFIDFRetriever(), LLMInterface())
    if documents:
        system.load_documents(documents)
    return system


def create_embedding_rag(documents: List[Document] = None, 
                        model: str = "all-MiniLM-L6-v2") -> RAGSystem:
    """Create an embedding-based RAG system"""
    system = RAGSystem(EmbeddingRetriever(model), LLMInterface())
    if documents:
        system.load_documents(documents)
    return system


def create_comparison_study(documents: List[Document]) -> RAGComparison:
    """Create a comparison study with both methods"""
    return RAGComparison(documents)