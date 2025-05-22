"""
Evaluation and experiment runner for RAG systems.
Provides metrics, comparison tools, and experiment orchestration.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from .rag_core import RAGSystem, TFIDFRetriever, EmbeddingRetriever, LLMInterface
from .datasets import DatasetLoader


class RAGMetrics:
    """
    Calculate performance metrics for RAG systems.
    Focuses on practical metrics useful for comparison.
    """
    
    @staticmethod
    def retrieval_success_rate(results: List[Dict]) -> float:
        """
        Calculate percentage of queries that retrieved at least one document.
        
        Args:
            results: List of RAG system results
            
        Returns:
            Success rate as percentage (0.0 to 1.0)
        """
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
    def calculate_summary_stats(results: List[Dict]) -> Dict[str, float]:
        """
        Calculate comprehensive summary statistics.
        
        Returns:
            Dictionary with all key metrics
        """
        return {
            "total_questions": len(results),
            "success_rate": RAGMetrics.retrieval_success_rate(results),
            "avg_response_time": RAGMetrics.average_response_time(results),
            "avg_similarity": RAGMetrics.average_similarity_score(results),
            "min_response_time": min((r.get('retrieval_time', 0) for r in results), default=0),
            "max_response_time": max((r.get('retrieval_time', 0) for r in results), default=0),
            "total_docs_retrieved": sum(r.get('docs_found', 0) for r in results)
        }


class ExperimentRunner:
    """
    Orchestrates RAG experiments and comparisons.
    Perfect for generating Medium article content and analysis.
    """
    
    def __init__(self, data_dir: str = "data", results_dir: str = "results"):
        """
        Initialize experiment runner.
        
        Args:
            data_dir: Directory containing datasets
            results_dir: Directory to save experiment results
        """
        self.data_loader = DatasetLoader(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Store experiment results for analysis
        self.experiment_history = []
    
    def create_rag_systems(self) -> Dict[str, RAGSystem]:
        """
        Create different RAG system configurations for comparison.
        
        Returns:
            Dictionary mapping method names to RAG systems
        """
        llm = LLMInterface()  # Shared LLM for fair comparison
        
        systems = {
            "tfidf": RAGSystem(TFIDFRetriever(), llm),
            "embeddings": RAGSystem(EmbeddingRetriever(), llm)
        }
        
        return systems
    
    def run_single_experiment(self, 
                            dataset_file: str,
                            test_questions_file: str = "test_questions.json",
                            top_k: int = 3) -> Dict[str, Any]:
        """
        Run a complete experiment comparing all RAG methods.
        
        Args:
            dataset_file: Name of Q&A dataset file
            test_questions_file: Name of test questions file
            top_k: Number of documents to retrieve
            
        Returns:
            Complete experiment results
        """
        print(f"🧪 Starting RAG Comparison Experiment")
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
                
                result = system.ask(question, top_k)
                method_results.append(result)
            
            # Calculate statistics
            stats = RAGMetrics.calculate_summary_stats(method_results)
            stats["total_experiment_time"] = time.time() - start_time
            
            method_stats[method_name] = stats
            all_results[method_name] = method_results
            
            # Print immediate results
            print(f"   ✅ Success Rate: {stats['success_rate']:.1%}")
            print(f"   ⏱️  Avg Time: {stats['avg_response_time']:.2f}s")
            print(f"   📊 Avg Similarity: {stats['avg_similarity']:.3f}")
        
        # Compile complete results
        experiment_results = {
            "metadata": {
                "dataset": dataset_file,
                "test_questions": test_questions_file,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_documents": len(documents),
                "total_test_questions": len(test_questions),
                "top_k": top_k
            },
            "summary_stats": method_stats,
            "detailed_results": all_results,
            "test_questions": test_questions
        }
        
        # Store in history
        self.experiment_history.append(experiment_results)
        
        # Print comparison summary
        self._print_comparison_summary(method_stats)
        
        return experiment_results
    
    def _print_comparison_summary(self, method_stats: Dict[str, Dict]):
        """Print a formatted comparison table"""
        print(f"\n📊 EXPERIMENT SUMMARY")
        print("=" * 70)
        print(f"{'Method':<12} {'Success':<8} {'Avg Time':<10} {'Similarity':<12} {'Total Time':<10}")
        print("-" * 70)
        
        for method, stats in method_stats.items():
            print(f"{method:<12} "
                  f"{stats['success_rate']:<8.1%} "
                  f"{stats['avg_response_time']:<10.2f}s "
                  f"{stats['avg_similarity']:<12.3f} "
                  f"{stats['total_experiment_time']:<10.1f}s")
        
        # Determine winner
        best_method = max(method_stats.keys(), 
                         key=lambda x: method_stats[x]['success_rate'])
        print(f"\n🏆 Best performing method: {best_method.upper()}")
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save experiment results to JSON file.
        
        Args:
            results: Experiment results dictionary
            filename: Optional filename, auto-generated if None
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"rag_experiment_{timestamp}.json"
        
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
        
        print(f"💾 Results saved to: {filepath}")
        return str(filepath)
    
    def quick_comparison(self, question: str, dataset_file: str = "ai_engineering_qa.json") -> Dict:
        """
        Quick comparison of methods on a single question.
        Useful for debugging and demonstration.
        
        Args:
            question: Single question to test
            dataset_file: Dataset to use for context
            
        Returns:
            Comparison results for the question
        """
        print(f"🔍 Quick comparison for: '{question}'")
        print("-" * 50)
        
        # Load documents
        documents = self.data_loader.load_qa_dataset(dataset_file)
        systems = self.create_rag_systems()
        
        results = {}
        
        for method_name, system in systems.items():
            system.load_documents(documents)
            result = system.ask(question)
            results[method_name] = result
            
            print(f"\n{method_name.upper()}:")
            print(f"  📊 Similarity: {result.get('avg_similarity', 0):.3f}")
            print(f"  📚 Docs found: {result.get('docs_found', 0)}")
            print(f"  ⏱️  Time: {result.get('retrieval_time', 0):.2f}s")
            print(f"  💬 Answer: {result.get('answer', '')[:100]}...")
        
        return results
    
    def analyze_method_strengths(self, results: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Analyze which method performs better on different types of questions.
        
        Args:
            results: Experiment results from run_single_experiment
            
        Returns:
            Dictionary showing strengths of each method
        """
        if "detailed_results" not in results:
            return {}
        
        analysis = {"tfidf_better": [], "embeddings_better": [], "tied": []}
        
        # Compare question by question
        questions = results["test_questions"]
        tfidf_results = results["detailed_results"].get("tfidf", [])
        embed_results = results["detailed_results"].get("embeddings", [])
        
        for i, question in enumerate(questions):
            if i >= len(tfidf_results) or i >= len(embed_results):
                continue
            
            tfidf_sim = tfidf_results[i].get("avg_similarity", 0)
            embed_sim = embed_results[i].get("avg_similarity", 0)
            
            if tfidf_sim > embed_sim * 1.1:  # 10% threshold
                analysis["tfidf_better"].append(question)
            elif embed_sim > tfidf_sim * 1.1:
                analysis["embeddings_better"].append(question)
            else:
                analysis["tied"].append(question)
        
        return analysis


# Convenience functions for easy usage
def run_quick_experiment(dataset: str = "ai_engineering_qa.json") -> Dict[str, Any]:
    """Run a quick experiment with default settings"""
    runner = ExperimentRunner()
    results = runner.run_single_experiment(dataset)
    
    if results:
        runner.save_results(results)
    
    return results


def compare_single_question(question: str) -> Dict:
    """Quick comparison of a single question"""
    runner = ExperimentRunner()
    return runner.quick_comparison(question)


if __name__ == "__main__":
    # Demo usage
    print("📊 RAG Evaluation Demo")
    print("=" * 30)
    
    # Run quick experiment
    try:
        results = run_quick_experiment()
        
        if results:
            print(f"\n🎯 Experiment completed!")
            print(f"📈 Methods compared: {list(results['summary_stats'].keys())}")
            print(f"📝 Results saved to 'results/' directory")
        
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        print("💡 Make sure you have data files in 'data/' directory")
        print("   Run: python -c 'from src.datasets import setup_sample_data; setup_sample_data()'")