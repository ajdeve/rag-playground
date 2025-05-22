"""
Dataset management for RAG experiments.
Loads Q&A data from JSON files and converts to Document objects.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from langchain.schema import Document


class DatasetLoader:
    """
    Handles loading and managing Q&A datasets from JSON files.
    Keeps data separate from code for better maintainability.
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize with data directory path"""
        self.data_dir = Path(data_dir)
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(exist_ok=True)
    
    def load_qa_dataset(self, filename: str) -> List[Document]:
        """
        Load Q&A dataset from JSON file and convert to Document objects.
        
        Args:
            filename: Name of JSON file (e.g., "ai_engineering_qa.json")
            
        Returns:
            List of Document objects ready for RAG indexing
            
        Expected JSON format:
        [
            {
                "id": "unique_id",
                "question": "What is...?",
                "answer": "The answer is...",
                "category": "optional_category",
                "tags": ["tag1", "tag2"]
            }
        ]
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            # Validate data structure
            if not isinstance(qa_data, list):
                raise ValueError("Dataset must be a list of Q&A objects")
            
            # Convert to Document objects
            documents = []
            for i, qa in enumerate(qa_data):
                # Validate required fields
                if not all(key in qa for key in ['id', 'question', 'answer']):
                    raise ValueError(f"Item {i} missing required fields (id, question, answer)")
                
                # Create document with Q&A content
                content = f"Q: {qa['question']}\nA: {qa['answer']}"
                
                # Preserve all metadata for retrieval and analysis
                metadata = {
                    "qa_id": qa["id"],  # This is the key field!
                    "question": qa["question"],
                    "category": qa.get("category", "general"),
                    "tags": qa.get("tags", []),
                    "source": filename
                }
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
                
                # Debug: Print first few documents to verify metadata
                if i < 3:
                    print(f"DEBUG Document {i+1} created:")
                    print(f"  - QA ID: {metadata['qa_id']}")
                    print(f"  - Question: {qa['question'][:50]}...")
                    print(f"  - Metadata: {metadata}")
            
            print(f"✅ Loaded {len(documents)} Q&A pairs from {filename}")
            return documents
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filename}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading {filename}: {e}")
    
    def load_test_questions(self, filename: str = "test_questions.json") -> List[str]:
        """
        Load test questions for evaluation.
        
        Args:
            filename: Name of test questions file
            
        Returns:
            List of test question strings
            
        Expected JSON format:
        {
            "dataset": "ai_engineering",
            "questions": [
                "What is RAG?",
                "How do embeddings work?"
            ]
        }
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Test questions file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            questions = data.get("questions", [])
            if not isinstance(questions, list):
                raise ValueError("Test questions must be a list")
            
            print(f"✅ Loaded {len(questions)} test questions from {filename}")
            return questions
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filename}: {e}")
    
    def list_available_datasets(self) -> List[str]:
        """List all available dataset files in the data directory"""
        json_files = list(self.data_dir.glob("*.json"))
        return [f.name for f in json_files if not f.name.startswith("test_")]
    
    def create_sample_dataset(self, filename: str, dataset_type: str = "ai_engineering"):
        """
        Create a sample dataset file if it doesn't exist.
        Useful for first-time setup.
        
        Args:
            filename: Name for the new dataset file
            dataset_type: Type of sample data to create
        """
        file_path = self.data_dir / filename
        
        if file_path.exists():
            print(f"⚠️  Dataset {filename} already exists, skipping creation")
            return
        
        if dataset_type == "ai_engineering":
            # Create minimal sample for testing
            sample_data = [
                {
                    "id": "rag_basics",
                    "question": "What is RAG in AI?",
                    "answer": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. It retrieves relevant documents and uses them as context for generating more accurate responses.",
                    "category": "basics",
                    "tags": ["rag", "ai", "retrieval"]
                },
                {
                    "id": "embedding_vs_tfidf",
                    "question": "What's the difference between embeddings and TF-IDF?",
                    "answer": "TF-IDF is keyword-based and fast but misses semantic meaning. Embeddings capture semantic relationships and understand context, but require more computation.",
                    "category": "comparison", 
                    "tags": ["embeddings", "tfidf", "comparison"]
                }
            ]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Write sample data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created sample dataset: {filename}")
    
    def get_dataset_info(self, filename: str) -> Dict[str, any]:
        """
        Get information about a dataset without loading all documents.
        
        Args:
            filename: Dataset filename
            
        Returns:
            Dictionary with dataset statistics
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Calculate statistics
        categories = set()
        all_tags = set()
        
        for item in data:
            if 'category' in item:
                categories.add(item['category'])
            if 'tags' in item:
                all_tags.update(item['tags'])
        
        return {
            "filename": filename,
            "total_items": len(data),
            "categories": list(categories),
            "unique_tags": list(all_tags),
            "has_categories": len(categories) > 0,
            "has_tags": len(all_tags) > 0
        }


# Convenience functions for easy usage
def load_ai_engineering_qa(data_dir: str = "data") -> List[Document]:
    """Quick function to load main AI engineering dataset"""
    loader = DatasetLoader(data_dir)
    return loader.load_qa_dataset("ai_engineering_qa.json")


def load_test_questions(data_dir: str = "data") -> List[str]:
    """Quick function to load test questions"""
    loader = DatasetLoader(data_dir)
    return loader.load_test_questions("test_questions.json")


def setup_sample_data(data_dir: str = "data"):
    """
    Setup sample datasets for first-time users.
    Creates both Q&A data and test questions.
    """
    loader = DatasetLoader(data_dir)
    
    # Create sample Q&A dataset
    loader.create_sample_dataset("ai_engineering_qa.json", "ai_engineering")
    
    # Create sample test questions
    test_data = {
        "dataset": "ai_engineering",
        "description": "Test questions for evaluating RAG system performance",
        "questions": [
            "What is RAG?",
            "How do embeddings compare to TF-IDF?",
            "What are the benefits of retrieval-augmented generation?",
            "When should I use semantic search vs keyword search?"
        ]
    }
    
    test_file = Path(data_dir) / "test_questions.json"
    if not test_file.exists():
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Created test questions: {test_file}")


if __name__ == "__main__":
    # Demo usage
    print("🗄️  Dataset Loader Demo")
    print("=" * 30)
    
    # Setup sample data
    setup_sample_data()
    
    # Load and show dataset info
    loader = DatasetLoader()
    
    try:
        documents = loader.load_qa_dataset("ai_engineering_qa.json")
        questions = loader.load_test_questions()
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Documents: {len(documents)}")
        print(f"   Test Questions: {len(questions)}")
        print(f"   Available Datasets: {loader.list_available_datasets()}")
        
    except FileNotFoundError:
        print("📁 Run setup_sample_data() first to create sample datasets")