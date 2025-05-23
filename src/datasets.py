"""
Dataset management for RAG experiments.
Loads Q&A data from JSON files and converts to Document objects.
FIXED: Now includes expected_answer in metadata for generation quality evaluation.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any  # Added Any import
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
                "expected_answer": "Expected response...",  # NEW: For generation evaluation
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
                    "expected_answer": qa.get("expected_answer"),  # FIXED: Include expected_answer
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
                    print(f"  - Expected Answer: {metadata['expected_answer'][:50] if metadata['expected_answer'] else 'None'}...")
                    print(f"  - Metadata keys: {list(metadata.keys())}")
            
            print(f"✅ Loaded {len(documents)} Q&A pairs from {filename}")
            return documents
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filename}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading {filename}: {e}")
    
    def load_test_questions_with_mapping(self, filename: str = "test_questions.json") -> Dict[str, Any]:
        """
        Load test questions with mapping information.
        
        Args:
            filename: Name of test questions file
            
        Returns:
            Dictionary containing questions, mapping, and metadata
            
        Expected JSON format:
        {
            "dataset": "ai_engineering",
            "questions": ["Question 1", "Question 2"],
            "question_mapping": {
                "Question 1": "qa_id_1",
                "Question 2": "qa_id_2"
            }
        }
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Test questions file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            questions = data.get("questions", [])
            question_mapping = data.get("question_mapping", {})
            
            if not isinstance(questions, list):
                raise ValueError("Test questions must be a list")
            
            if not isinstance(question_mapping, dict):
                raise ValueError("Question mapping must be a dictionary")
            
            print(f"✅ Loaded {len(questions)} test questions from {filename}")
            print(f"✅ Loaded {len(question_mapping)} question mappings")
            
            return {
                "questions": questions,
                "question_mapping": question_mapping,
                "dataset": data.get("dataset", "unknown"),
                "description": data.get("description", ""),
                "version": data.get("version", "1.0")
            }
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filename}: {e}")
    
    def load_test_questions(self, filename: str = "test_questions.json") -> List[str]:
        """
        Load test questions for evaluation (legacy method for compatibility).
        
        Args:
            filename: Name of test questions file
            
        Returns:
            List of test question strings
        """
        try:
            data = self.load_test_questions_with_mapping(filename)
            return data.get("questions", [])
        except Exception:
            # Fallback to old format
            file_path = self.data_dir / filename
            
            if not file_path.exists():
                raise FileNotFoundError(f"Test questions file not found: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                questions = data.get("questions", [])
                if not isinstance(questions, list):
                    raise ValueError("Test questions must be a list")
                
                print(f"✅ Loaded {len(questions)} test questions from {filename} (legacy format)")
                return questions
                
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {filename}: {e}")
    
    def list_available_datasets(self) -> List[str]:
        """List all available dataset files in the data directory"""
        json_files = list(self.data_dir.glob("*.json"))
        return [f.name for f in json_files if not f.name.startswith("test_")]
    
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
        has_expected_answers = 0
        
        for item in data:
            if 'category' in item:
                categories.add(item['category'])
            if 'tags' in item:
                all_tags.update(item['tags'])
            if 'expected_answer' in item and item['expected_answer']:
                has_expected_answers += 1
        
        return {
            "filename": filename,
            "total_items": len(data),
            "categories": list(categories),
            "unique_tags": list(all_tags),
            "has_categories": len(categories) > 0,
            "has_tags": len(all_tags) > 0,
            "expected_answers_count": has_expected_answers,
            "generation_eval_ready": has_expected_answers > 0
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


if __name__ == "__main__":
    # Demo usage
    print("🗄️  Dataset Loader Demo (Fixed Version)")
    print("=" * 40)
    
    # Load and show dataset info
    loader = DatasetLoader()
    
    try:
        documents = loader.load_qa_dataset("ai_engineering_qa.json")
        questions = loader.load_test_questions()
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Documents: {len(documents)}")
        print(f"   Test Questions: {len(questions)}")
        print(f"   Available Datasets: {loader.list_available_datasets()}")
        
        # Check if expected answers are properly loaded
        docs_with_expected = sum(1 for doc in documents if doc.metadata.get('expected_answer'))
        print(f"   Documents with expected answers: {docs_with_expected}/{len(documents)}")
        
        if docs_with_expected > 0:
            print("✅ Generation quality evaluation ready!")
        else:
            print("⚠️  No expected answers found - check your dataset")
        
    except FileNotFoundError:
        print("📁 Run the setup script first to create sample datasets")