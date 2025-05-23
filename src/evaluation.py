"""
Evaluation and experiment runner for RAG systems.
Provides metrics, comparison tools, and experiment orchestration.
Enhanced with simplified generation quality evaluation.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .rag_core import RAGSystem, TFIDFRetriever, EmbeddingRetriever, LLMInterface
from .datasets import DatasetLoader


class GenerationEvaluator:
    """
    Simple evaluation of generated answer quality in RAG systems.
    Focuses on answer similarity to expected responses.
    """
    
    def __init__(self):
        """Initialize with sentence transformer for semantic similarity"""
        # Use same model as embeddings for consistency
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def evaluate_answer_similarity(self, generated_answer: str, expected_answer: str) -> float:
        """
        Calculate semantic similarity between generated and expected answers.
        
        Args:
            generated_answer: Answer produced by RAG system
            expected_answer: Ground truth answer from dataset
            
        Returns:
            Similarity score between 0 and 1
        """
        if not generated_answer or not expected_answer:
            return 0.0
        
        # Encode both answers
        embeddings = self.similarity_model.encode([generated_answer, expected_answer])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
        return float(similarity)


class RAGMetrics:
    """
    Calculate performance metrics for RAG systems.
    Enhanced with simplified generation quality metrics.
    """
    
    @staticmethod
    def retrieval_success_rate(results: List[Dict]) -> float:
        """Calculate percentage of queries that retrieved at least one document"""
        if not results:
            return 0.0
        
        successful = sum(1 for r in results if r.get('docs_found', 0) > 0)
        return successful / len(results)
    
    @staticmethod
    def average_response_time(results: List[Dict]) -> float:
        """Calculate average response time across all queries"""
        if not results:
            return 0.0
        
        times = [r.get('retrieval_time', 0) for r in results]
        return np.mean(times)
    
    @staticmethod
    def average_similarity_score(results: List[Dict]) -> float:
        """Calculate average similarity score for successful retrievals"""
        similarities = []
        
        for result in results:
            if result.get('docs_found', 0) > 0:
                similarities.append(result.get('avg_similarity', 0))
        
        return np.mean(similarities) if similarities else 0.0
    
    @staticmethod
    def average_generation_quality(results: List[Dict]) -> Dict[str, float]:
        """Calculate average generation quality metrics (simplified)"""
        answer_similarities = [r.get('answer_similarity', 0) for r in results if 'answer_similarity' in r]
        
        return {
            "avg_answer_similarity": np.mean(answer_similarities) if answer_similarities else 0.0,
            "answer_evaluation_count": len(answer_similarities)
        }
    
    @staticmethod
    def calculate_summary_stats(results: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive summary statistics including generation quality"""
        base_stats = {
            "total_questions": len(results),
            "success_rate": RAGMetrics.retrieval_success_rate(results),
            "avg_response_time": RAGMetrics.average_response_time(results),
            "avg_similarity": RAGMetrics.average_similarity_score(results),
            "min_response_time": min((r.get('retrieval_time', 0) for r in results), default=0),
            "max_response_time": max((r.get('retrieval_time', 0) for r in results), default=0),
            "total_docs_retrieved": sum(r.get('docs_found', 0) for r in results)
        }
        
        # Add generation quality metrics
        generation_stats = RAGMetrics.average_generation_quality(results)
        base_stats.update(generation_stats)
        
        return base_stats


class ExperimentRunner:
    """
    Orchestrates RAG experiments and comparisons with generation quality evaluation.
    """
    
    def __init__(self, data_dir: str = "data", results_dir: str = "results"):
        """Initialize experiment runner with generation evaluator"""
        self.data_loader = DatasetLoader(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize generation evaluator
        self.generation_evaluator = GenerationEvaluator()
        
        # Store experiment results for analysis
        self.experiment_history = []
    
    def create_rag_systems(self) -> Dict[str, RAGSystem]:
        """Create different RAG system configurations for comparison"""
        llm = LLMInterface()  # Shared LLM for fair comparison
        
        systems = {
            "tfidf": RAGSystem(TFIDFRetriever(), llm),
            "embeddings": RAGSystem(EmbeddingRetriever(), llm)
        }
        
        return systems
    
    def evaluate_single_answer(self, result: Dict, expected_answer: str, question: str) -> Dict:
        """
        Evaluate a single generated answer against expected answer (simplified).
        
        Args:
            result: RAG system result dictionary
            expected_answer: Ground truth answer
            question: Original question
            
        Returns:
            Enhanced result with generation quality metrics
        """
        generated_answer = result.get('answer', '')
        
        # Calculate simple answer similarity
        answer_similarity = self.generation_evaluator.evaluate_answer_similarity(
            generated_answer, expected_answer
        )
        
        # Add generation metrics to result (simplified)
        enhanced_result = result.copy()
        enhanced_result.update({
            "expected_answer": expected_answer,
            "answer_similarity": answer_similarity
        })
        
        return enhanced_result
    
    def run_single_experiment(self, 
                            dataset_file: str,
                            test_questions_file: str = "test_questions.json",
                            top_k: int = 3) -> Dict[str, Any]:
        """
        Run a complete experiment comparing all RAG methods with generation quality evaluation.
        """
        print(f"🧪 Starting Enhanced RAG Comparison Experiment")
        print(f"📄 Dataset: {dataset_file}")
        print(f"❓ Test questions: {test_questions_file}")
        print("=" * 50)
        
        # Load data
        try:
            documents = self.data_loader.load_qa_dataset(dataset_file)
            test_questions = self.data_loader.load_test_questions(test_questions_file)
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            return {}
        
        # Create RAG systems
        systems = self.create_rag_systems()
        
        # Run experiments for each method
        all_results = {}
        method_stats = {}
        
        for method_name, system in systems.items():
            print(f"\n🔧 Testing {method_name.upper()}...")
            
            # Load documents into system
            system.load_documents(documents)
            
            # Test all questions
            start_time = time.time()
            method_results = []
            
            for i, question in enumerate(test_questions, 1):
                print(f"   Question {i}/{len(test_questions)}: {question[:50]}...")
                
                # Get RAG system result
                result = system.ask(question, top_k)
                
                # Find expected answer (try to match by question content)
                expected_answer = self._find_expected_answer(question, documents)
                
                if expected_answer:
                    # Evaluate generation quality
                    enhanced_result = self.evaluate_single_answer(result, expected_answer, question)
                    method_results.append(enhanced_result)
                else:
                    # No expected answer found, just use basic result
                    method_results.append(result)
            
            # Calculate statistics including generation quality
            stats = RAGMetrics.calculate_summary_stats(method_results)
            stats["total_experiment_time"] = time.time() - start_time
            
            method_stats[method_name] = stats
            all_results[method_name] = method_results
            
            # Print immediate results (simplified)
            print(f"   ✅ Retrieval Success: {stats['success_rate']:.1%}")
            print(f"   ⏱️  Avg Time: {stats['avg_response_time']:.2f}s")
            print(f"   📊 Retrieval Quality: {stats['avg_similarity']:.3f}")
            if stats.get('avg_answer_similarity', 0) > 0:
                print(f"   🎯 Answer Quality: {stats.get('avg_answer_similarity', 0):.3f}")
        
        # Compile complete results
        experiment_results = {
            "metadata": {
                "dataset": dataset_file,
                "test_questions": test_questions_file,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_documents": len(documents),
                "total_test_questions": len(test_questions),
                "top_k": top_k,
                "evaluation_enhanced": True
            },
            "summary_stats": method_stats,
            "detailed_results": all_results,
            "test_questions": test_questions
        }
        
        # Store in history
        self.experiment_history.append(experiment_results)
        
        # Print comparison summary
        self._print_enhanced_comparison_summary(method_stats)
        
        return experiment_results
    
    def _find_expected_answer(self, question: str, documents: List) -> Optional[str]:
        """Find expected answer for a question by matching against document metadata"""
        # First try exact question match
        for doc in documents:
            doc_question = doc.metadata.get('question', '')
            if question.strip().lower() == doc_question.strip().lower():
                expected = doc.metadata.get('expected_answer')
                if expected:
                    return expected
        
        # Then try partial match
        for doc in documents:
            doc_question = doc.metadata.get('question', '')
            # Check if key words match
            question_words = set(question.lower().split())
            doc_words = set(doc_question.lower().split())
            
            # If there's significant overlap, consider it a match
            overlap = len(question_words.intersection(doc_words))
            if overlap >= 3:  # At least 3 words in common
                expected = doc.metadata.get('expected_answer')
                if expected:
                    return expected
        
        return None
    
    def _print_enhanced_comparison_summary(self, method_stats: Dict[str, Dict]):
        """Print a simplified comparison table with essential metrics"""
        print(f"\n📊 EXPERIMENT SUMMARY")
        print("=" * 70)
        print(f"{'Method':<12} {'Success':<8} {'Avg Time':<10} {'Retrieval':<10} {'Answer':<10}")
        print(f"{'':12} {'Rate':8} {'(s)':10} {'Quality':10} {'Quality':10}")
        print("-" * 70)
        
        for method, stats in method_stats.items():
            print(f"{method:<12} "
                  f"{stats['success_rate']:<8.1%} "
                  f"{stats['avg_response_time']:<10.2f} "
                  f"{stats['avg_similarity']:<10.3f} "
                  f"{stats.get('avg_answer_similarity', 0):<10.3f}")
        
        # Determine winners in different categories
        best_retrieval = max(method_stats.keys(), key=lambda x: method_stats[x]['avg_similarity'])
        best_generation = max(method_stats.keys(), key=lambda x: method_stats[x].get('avg_answer_similarity', 0))
        fastest = min(method_stats.keys(), key=lambda x: method_stats[x]['avg_response_time'])
        
        print(f"\n🏆 Winners:")
        print(f"   Best Retrieval: {best_retrieval.upper()}")
        print(f"   Best Answers: {best_generation.upper()}")
        print(f"   Fastest: {fastest.upper()}")
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save experiment results to JSON file with generation quality data"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_rag_experiment_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Save with proper formatting
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=convert_types, ensure_ascii=False)
        
        print(f"💾 Enhanced results saved to: {filepath}")
        return str(filepath)
    
    def quick_comparison(self, question: str, dataset_file: str = "ai_engineering_qa.json") -> Dict:
        """Quick comparison with generation quality evaluation"""
        print(f"🔍 Enhanced comparison for: '{question}'")
        print("-" * 50)
        
        # Load documents
        documents = self.data_loader.load_qa_dataset(dataset_file)
        systems = self.create_rag_systems()
        
        results = {}
        
        for method_name, system in systems.items():
            system.load_documents(documents)
            result = system.ask(question)
            
            # Try to find expected answer
            expected_answer = self._find_expected_answer(question, documents)
            
            if expected_answer:
                enhanced_result = self.evaluate_single_answer(result, expected_answer, question)
                results[method_name] = enhanced_result
                
                print(f"\n{method_name.upper()}:")
                print(f"  📊 Retrieval Quality: {result.get('avg_similarity', 0):.3f}")
                print(f"  📚 Docs found: {result.get('docs_found', 0)}")
                print(f"  ⏱️  Time: {result.get('retrieval_time', 0):.2f}s")
                print(f"  🎯 Answer Quality: {enhanced_result.get('answer_similarity', 0):.3f}")
                print(f"  💬 Answer: {result.get('answer', '')[:100]}...")
            else:
                results[method_name] = result
                print(f"\n{method_name.upper()} (No expected answer for comparison):")
                print(f"  📊 Similarity: {result.get('avg_similarity', 0):.3f}")
                print(f"  📚 Docs found: {result.get('docs_found', 0)}")
                print(f"  ⏱️  Time: {result.get('retrieval_time', 0):.2f}s")
                print(f"  💬 Answer: {result.get('answer', '')[:100]}...")
        
        return results


# Enhanced convenience functions
def run_enhanced_experiment(dataset: str = "ai_engineering_qa.json") -> Dict[str, Any]:
    """Run an enhanced experiment with generation quality evaluation"""
    runner = ExperimentRunner()
    results = runner.run_single_experiment(dataset)
    
    if results:
        runner.save_results(results)
    
    return results


def compare_single_question_enhanced(question: str) -> Dict:
    """Enhanced comparison of a single question with generation quality metrics"""
    runner = ExperimentRunner()
    return runner.quick_comparison(question)


if __name__ == "__main__":
    # Demo usage with enhanced evaluation
    print("📊 Enhanced RAG Evaluation Demo")
    print("=" * 40)
    
    # Run enhanced experiment
    try:
        results = run_enhanced_experiment()
        
        if results:
            print(f"\n🎯 Enhanced experiment completed!")
            print(f"📈 Methods compared: {list(results['summary_stats'].keys())}")
            print(f"📊 Generation quality metrics included!")
            print(f"📝 Results saved to 'results/' directory")
        
    except Exception as e:
        print(f"❌ Error running enhanced experiment: {e}")
        print("💡 Make sure you have data files in 'data/' directory")
        print("   Run: python -c 'from src.datasets import setup_sample_data; setup_sample_data()'")