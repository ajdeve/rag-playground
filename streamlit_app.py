"""
Interactive Streamlit app for RAG system demonstration and experimentation.
Clean, production-ready code with proper error handling and documentation.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
from typing import Dict, List, Any, Optional, Tuple

# Import our RAG components
try:
    from src.rag_core import RAGSystem, TFIDFRetriever, EmbeddingRetriever, LLMInterface
    from src.datasets import DatasetLoader
    from src.evaluation import ExperimentRunner, RAGMetrics, GenerationEvaluator
except ImportError as e:
    # Try alternative import paths
    try:
        import sys
        sys.path.append('.')
        from rag_core import RAGSystem, TFIDFRetriever, EmbeddingRetriever, LLMInterface
        from datasets import DatasetLoader
        from evaluation import ExperimentRunner, RAGMetrics, GenerationEvaluator
    except ImportError:
        st.error(f"❌ Cannot import RAG modules: {e}")
        st.error("Make sure you're running from the project root directory and all dependencies are installed.")
        st.stop()

# Configuration constants
CONFIG = {
    "PAGE_TITLE": "RAG Learning Lab",
    "PAGE_ICON": "🤖",
    "LAYOUT": "wide",
    "MAX_SAMPLE_QUESTIONS": 6,
    "DEFAULT_TOP_K": 3,
    "HIGH_QUALITY_THRESHOLD": 0.7,
    "MEDIUM_QUALITY_THRESHOLD": 0.4,
    "TARGET_LATENCY_GOOD": 2.0,
    "TARGET_LATENCY_REALTIME": 0.5,
}

# Page configuration
st.set_page_config(
    page_title=CONFIG["PAGE_TITLE"],
    page_icon=CONFIG["PAGE_ICON"],
    layout=CONFIG["LAYOUT"],
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .comparison-box {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_datasets() -> Tuple[Optional[List], Optional[List], Optional[Dict], Optional[str]]:
    """Load datasets with caching for performance."""
    try:
        loader = DatasetLoader()
        documents = loader.load_qa_dataset("ai_engineering_qa.json")
        
        # Try to load test questions with mapping, fall back to simple loading
        try:
            test_data = loader.load_test_questions_with_mapping("test_questions.json")
            test_questions = test_data.get("questions", [])
            question_mapping = test_data.get("question_mapping", {})
        except AttributeError:
            # Fallback: load questions without mapping
            test_questions = loader.load_test_questions("test_questions.json")
            question_mapping = {}
            
            # Try to load mapping manually if file exists
            try:
                import json
                from pathlib import Path
                test_file = Path("data/test_questions.json")
                if test_file.exists():
                    with open(test_file, 'r') as f:
                        data = json.load(f)
                        question_mapping = data.get("question_mapping", {})
            except:
                question_mapping = {}
        
        return documents, test_questions, question_mapping, None
    except FileNotFoundError as e:
        return None, None, None, f"Dataset files not found: {e}"
    except Exception as e:
        return None, None, None, f"Error loading datasets: {e}"


@st.cache_resource
def initialize_rag_systems() -> Tuple[Optional[Dict[str, RAGSystem]], Optional[str]]:
    """Initialize RAG systems with caching."""
    documents, _, _, error = load_datasets()
    
    if error or not documents:
        return None, error or "No documents available"
    
    try:
        llm = LLMInterface()
        systems = {
            "TF-IDF": RAGSystem(TFIDFRetriever(), llm),
            "Embeddings": RAGSystem(EmbeddingRetriever(), llm)
        }
        
        for system in systems.values():
            system.load_documents(documents)
        
        return systems, None
        
    except Exception as e:
        return None, f"Error initializing RAG systems: {e}"


def get_quality_label(score: float) -> str:
    """Get quality label based on score threshold."""
    if score > CONFIG["HIGH_QUALITY_THRESHOLD"]:
        return "High"
    elif score > CONFIG["MEDIUM_QUALITY_THRESHOLD"]:
        return "Medium"
    else:
        return "Low"


def get_quality_indicator(score: float) -> str:
    """Get visual quality indicator based on score."""
    if score > CONFIG["HIGH_QUALITY_THRESHOLD"]:
        return "●"
    elif score > CONFIG["MEDIUM_QUALITY_THRESHOLD"]:
        return "◐"
    else:
        return "○"


def find_expected_answer_by_mapping(question: str, documents: List, question_mapping: Dict) -> Optional[str]:
    """Find expected answer using the question mapping from test_questions.json."""
    if not documents or not question_mapping:
        return None
    
    # First try direct mapping lookup
    qa_id = question_mapping.get(question)
    if qa_id:
        for doc in documents:
            if doc.metadata.get('qa_id') == qa_id:
                return doc.metadata.get('expected_answer')
    
    return None


def find_expected_answer_fallback(question: str, documents: List) -> Optional[str]:
    """Fallback method for finding expected answers when mapping fails."""
    if not documents:
        return None
        
    question_lower = question.lower().strip()
    
    # Try exact matching first
    for doc in documents:
        doc_question = doc.metadata.get('question', '').lower().strip()
        if question_lower == doc_question:
            return doc.metadata.get('expected_answer')
    
    # Enhanced keyword matching with concept clustering
    question_keywords = set(question_lower.replace('?', '').replace(',', '').split())
    
    # Remove common words to focus on meaningful keywords
    common_words = {'i', 'my', 'is', 'do', 'how', 'what', 'when', 'should', 'the', 'a', 'an', 'and', 'or', 'to', 'for', 'of', 'in', 'on', 'with', 'vs'}
    question_keywords = question_keywords - common_words
    
    best_match = None
    best_score = 0
    
    for doc in documents:
        doc_question = doc.metadata.get('question', '').lower().strip()
        doc_keywords = set(doc_question.replace('?', '').replace(',', '').split()) - common_words
        
        if len(question_keywords) > 0 and len(doc_keywords) > 0:
            # Calculate overlap ratio
            overlap = question_keywords.intersection(doc_keywords)
            overlap_ratio = len(overlap) / len(question_keywords.union(doc_keywords))
            
            # Check for key concept matches
            key_concepts = {
                'rag': ['rag', 'retrieval'],
                'embedding': ['embedding', 'embeddings'],
                'fine-tune': ['fine-tuning', 'finetune', 'fine-tune'],
                'chunk': ['chunk', 'chunking'],
                'evaluation': ['evaluate', 'evaluation', 'working'],
                'cost': ['cost', 'costs', 'pricing'],
                'hallucination': ['hallucinate', 'hallucination', 'making'],
                'model': ['model', 'models'],
                'database': ['database', 'vector'],
                'speed': ['faster', 'speed', 'latency', 'optimize'],
                'prompt': ['prompt', 'prompts', 'context']
            }
            
            concept_matches = 0
            for concept, terms in key_concepts.items():
                if any(term in question_keywords for term in terms) and any(term in doc_keywords for term in terms):
                    concept_matches += 1
            
            # Boost score for concept matches
            final_score = overlap_ratio + (concept_matches * 0.2)
            
            if final_score > best_score and final_score > 0.3:  # Minimum threshold
                best_score = final_score
                best_match = doc.metadata.get('expected_answer')
    
    return best_match


def find_expected_answer(question: str, documents: List, question_mapping: Dict = None) -> Optional[str]:
    """
    Find expected answer using mapping first, then fallback methods.
    
    Args:
        question: The input question
        documents: List of document objects
        question_mapping: Optional mapping from test_questions.json
    
    Returns:
        Expected answer string if found, None otherwise
    """
    # Try mapping-based lookup first (most reliable)
    if question_mapping:
        expected = find_expected_answer_by_mapping(question, documents, question_mapping)
        if expected:
            return expected
    
    # Fallback to keyword/semantic matching
    return find_expected_answer_fallback(question, documents)


def calculate_generation_quality(result: Dict, expected_answer: str) -> float:
    """Calculate generation quality score between expected and generated answers."""
    try:
        gen_evaluator = GenerationEvaluator()
        answer_similarity = gen_evaluator.evaluate_answer_similarity(
            result.get('answer', ''), expected_answer
        )
        return answer_similarity
    except Exception as e:
        st.warning(f"Could not calculate generation quality: {e}")
        return 0.0


def process_rag_query(system: RAGSystem, question: str, top_k: int, expected_answer: Optional[str] = None) -> Dict:
    """Process a single RAG query with error handling."""
    try:
        result = system.ask(question, top_k)
        
        if expected_answer:
            answer_similarity = calculate_generation_quality(result, expected_answer)
            result['expected_answer'] = expected_answer
            result['answer_similarity'] = answer_similarity
        
        return result
        
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return {
            'answer': f'Error: {e}',
            'docs_found': 0,
            'retrieval_time': 0,
            'avg_similarity': 0,
            'context_length': 0,
            'similarity_scores': [],
            'retrieved_qa_ids': []
        }


def create_comparison_chart(results: Dict[str, Dict]) -> go.Figure:
    """Create a comparison chart for RAG methods."""
    methods = list(results.keys())
    
    fig = go.Figure()
    
    for method in methods:
        values = [
            results[method].get('success_rate', 0) * 100,
            results[method].get('avg_response_time', 0),
            results[method].get('avg_similarity', 0) * 100
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=['Success Rate (%)', 'Response Time (s)', 'Similarity Score (%)'],
            fill='toself',
            name=method,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True,
        title="RAG Methods Comparison",
        height=400
    )
    
    return fig


def display_overview_metrics(results: Dict[str, Dict]) -> None:
    """Display overview metrics for all methods."""
    st.metric("Methods Tested", len(results))


def display_method_comparison(results: Dict[str, Dict]) -> None:
    """Display comparison insights between methods."""
    if len(results) != 2:
        return
        
    methods = list(results.keys())
    method1, method2 = methods[0], methods[1]
    result1, result2 = results[method1], results[method2]
    
    better_retrieval = method1 if result1.get('avg_similarity', 0) > result2.get('avg_similarity', 0) else method2
    faster_method = method1 if result1.get('retrieval_time', 0) < result2.get('retrieval_time', 0) else method2
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Better Document Retrieval:** {better_retrieval} found more relevant documents")
    with col2:
        st.info(f"**Faster Method:** {faster_method} responded quicker")
    
    has_gen_quality = result1.get('answer_similarity') is not None and result2.get('answer_similarity') is not None
    if has_gen_quality:
        better_generation = method1 if result1.get('answer_similarity', 0) > result2.get('answer_similarity', 0) else method2
        st.success(f"**Better Answer Generation:** {better_generation} produced more accurate response")


def display_method_metrics(result: Dict[str, Any]) -> None:
    """Display metrics for a single method."""
    has_generation_quality = result.get('answer_similarity') is not None
    
    if has_generation_quality:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Time", f"{result.get('retrieval_time', 0):.2f}s")
        with col2:
            st.metric("Docs Found", result.get('docs_found', 0))
        with col3:
            st.metric("Doc Similarity", f"{result.get('avg_similarity', 0):.3f}")
        with col4:
            st.metric("Generation Quality", f"{result.get('answer_similarity', 0):.3f}")
        with col5:
            st.metric("Context Length", f"{result.get('context_length', 0)} chars")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Time", f"{result.get('retrieval_time', 0):.2f}s")
        with col2:
            st.metric("Docs Found", result.get('docs_found', 0))
        with col3:
            st.metric("Doc Similarity", f"{result.get('avg_similarity', 0):.3f}")
        with col4:
            st.metric("Context Length", f"{result.get('context_length', 0)} chars")


def display_answer_section(result: Dict[str, Any]) -> None:
    """Display the generated answer."""
    answer = result.get('answer', 'No answer generated')
    
    st.write("**Answer:**")
    st.write(answer)
    
    if result.get('expected_answer'):
        with st.expander("View Expected Answer"):
            st.write("**Expected Answer:**")
            st.write(result['expected_answer'])
            if result.get('answer_similarity') is not None:
                similarity_score = result.get('answer_similarity', 0)
                quality_label = get_quality_label(similarity_score)
                st.write(f"**Generation Quality**: {similarity_score:.3f} ({quality_label})")


def display_generation_quality_summary(result: Dict[str, Any]) -> None:
    """Display generation quality summary if available."""
    if result.get('answer_similarity') is None:
        return
        
    st.markdown("### Generation Quality Summary")
    generation_score = result.get('answer_similarity', 0)
    generation_label = get_quality_label(generation_score)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Answer Similarity", f"{generation_score:.3f}")
    with col2:
        st.metric("Quality Level", generation_label)
    with col3:
        expected_length = len(result.get('expected_answer', ''))
        generated_length = len(result.get('answer', ''))
        st.metric("Length Ratio", f"{generated_length}/{expected_length}")
    
    st.info("""
    **Generation Quality**: Measures how similar the LLM's generated answer is to the expected answer using cosine similarity.
    - **High (>0.7)**: Generated answer closely matches expected content
    - **Medium (0.4-0.7)**: Partial alignment with expected answer  
    - **Low (<0.4)**: Generated answer differs significantly from expected
    """)


def find_document_details(documents: List, qa_id: str, doc_index: int) -> Tuple[str, str, str, List[str]]:
    """Find document details by QA ID or index."""
    doc_content = "Content not available"
    doc_question = "Question not found"
    doc_category = "Unknown"
    doc_tags = []
    
    if qa_id != f"document_{doc_index}":
        for doc in documents:
            if doc.metadata.get('qa_id') == qa_id:
                doc_content = doc.page_content
                doc_question = doc.metadata.get('question', 'Unknown question')
                doc_category = doc.metadata.get('category', 'general')
                doc_tags = doc.metadata.get('tags', [])
                break
    else:
        if doc_index-1 < len(documents):
            doc = documents[doc_index-1]
            doc_content = doc.page_content
            doc_question = doc.metadata.get('question', 'Unknown question')
            doc_category = doc.metadata.get('category', 'general')
            doc_tags = doc.metadata.get('tags', [])
    
    return doc_content, doc_question, doc_category, doc_tags


def get_retrieval_explanation(method: str) -> str:
    """Get explanation for why a method selected a document."""
    if "tfidf" in method.lower():
        return "Shares important keywords with your query"
    else:
        return "Semantically similar meaning to your query"


def display_method_explainer(method: str) -> None:
    """Display explainer for TF-IDF method."""
    if "tfidf" not in method.lower():
        return
        
    with st.expander(f"How {method} Works", expanded=False):
        st.write("""
        **TF-IDF (Term Frequency-Inverse Document Frequency):**
        - **Tokenization**: Splits text into individual words and 2-word phrases
        - **TF (Term Frequency)**: Counts how often each word appears in each document
        - **IDF (Inverse Document Frequency)**: Calculates how rare/important each word is across all documents
        - **Weighting**: Multiplies TF × IDF to give higher scores to important, distinctive words
        - **Similarity**: Uses cosine similarity to measure overlap between query and document vectors
        - **Speed**: Very fast since it's just mathematical operations on sparse matrices
        
        **Strengths**: Fast, interpretable, good for exact keyword matches
        **Weaknesses**: Misses synonyms and semantic meaning (e.g., "car" ≠ "automobile")
        """)


def display_retrieved_documents(result: Dict[str, Any], documents: List, method: str) -> None:
    """Display detailed information about retrieved documents."""
    scores = result.get('similarity_scores', [])
    qa_ids = result.get('retrieved_qa_ids', [])
    
    if not scores:
        return
    
    with st.expander("View Retrieved Documents & Similarity Scores", expanded=False):
        if not documents:
            st.warning("Document details not available - documents list is empty")
            for i, score in enumerate(scores, 1):
                qa_id = qa_ids[i-1] if i-1 < len(qa_ids) else f"unknown_{i}"
                st.write(f"**Document {i}**: {qa_id} (Similarity: {score:.3f})")
            return
        
        for i, score in enumerate(scores, 1):
            qa_id = qa_ids[i-1] if i-1 < len(qa_ids) else f"document_{i}"
            
            doc_content, doc_question, doc_category, doc_tags = find_document_details(documents, qa_id, i)
            
            relevance_indicator = get_quality_indicator(score)
            score_label = f"{get_quality_label(score)} Relevance"
            
            st.markdown(f"""
            ### Document {i}: `{qa_id}`
            **{relevance_indicator} Similarity Score: {score:.3f}** ({score_label})
            - **Category:** `{doc_category}`
            - **Tags:** {', '.join(f'`{tag}`' for tag in doc_tags) if doc_tags else 'None'}
            - **Original Question:** *{doc_question}*
            """)
            
            st.markdown("**Retrieved Content:**")
            st.code(doc_content, language="text")
            
            explanation = get_retrieval_explanation(method)
            st.info(f"**Why {method} selected this**: {explanation}")
            
            if i < len(scores):
                st.markdown("---")
        
        display_generation_quality_summary(result)


def display_method_result(method: str, result: Dict[str, Any], documents: List) -> None:
    """Display complete result for a single method."""
    model_info = " (all-MiniLM-L6-v2)" if "embedding" in method.lower() else ""
    
    st.markdown(f"""
    <div class="comparison-box">
        <h4>{method}{model_info}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    display_method_metrics(result)
    display_method_explainer(method)
    display_answer_section(result)
    
    if result.get('docs_found', 0) > 0:
        qa_ids = result.get('retrieved_qa_ids', [])
        
        if qa_ids:
            sources_text = ', '.join(qa_ids)
            st.write(f"**Sources Found**: `{sources_text}`")
        else:
            st.write(f"**Sources Found**: {result.get('docs_found', 0)} documents (IDs not available)")
        
        display_retrieved_documents(result, documents, method)
    else:
        st.warning("No relevant documents found - try rephrasing your question")


def display_single_question_results(question: str, results: Dict[str, Dict]) -> None:
    """Display results for single question comparison with full explainability."""
    st.subheader(f"📋 Results for: *{question}*")
    
    documents, _, _, _ = load_datasets()
    
    display_overview_metrics(results)
    
    st.subheader("Understanding the Comparison")
    display_method_comparison(results)
    
    if len(results) == 2:
        col1, col2 = st.columns(2)
        methods = list(results.keys())
        
        for i, (method, result) in enumerate(results.items()):
            with col1 if i == 0 else col2:
                display_method_result(method, result, documents)
    else:
        for method, result in results.items():
            display_method_result(method, result, documents)


def initialize_session_state() -> None:
    """Initialize session state variables."""
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""


def display_sample_questions(test_questions: List[str], question_mapping: Dict = None) -> None:
    """Display clickable sample questions."""
    if not test_questions:
        return
        
    st.write("**Click a sample question to try:**")
    
    sample_cols = st.columns(3)
    max_questions = min(CONFIG["MAX_SAMPLE_QUESTIONS"], len(test_questions))
    
    for i, sample_q in enumerate(test_questions[:max_questions]):
        with sample_cols[i % 3]:
            # Simple button without indicators
            button_text = f"{sample_q[:35]}..."
            
            if st.button(button_text, key=f"sample_{i}", help=sample_q):
                st.session_state.current_question = sample_q
                st.rerun()


def process_single_question(systems: Dict[str, RAGSystem], selected_methods: List[str], question: str, top_k: int) -> None:
    """Process a single question through selected RAG methods."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    documents, _, question_mapping, _ = load_datasets()
    expected_answer = find_expected_answer(question, documents, question_mapping)
    
    results = {}
    
    for i, method_name in enumerate(selected_methods):
        status_text.text(f"Processing with {method_name}...")
        progress_bar.progress((i + 1) / len(selected_methods))
        
        system = systems[method_name]
        result = process_rag_query(system, question, top_k, expected_answer)
        results[method_name] = result
    
    progress_bar.empty()
    status_text.empty()
    
    display_single_question_results(question, results)


def single_question_mode(systems: Dict[str, RAGSystem], selected_methods: List[str], top_k: int) -> None:
    """Single question comparison interface."""
    st.header("Single Question Analysis")
    st.write("Ask a question and see how different RAG methods respond!")
    
    documents, test_questions, question_mapping, error = load_datasets()
    
    initialize_session_state()
    
    # Get questions to display - prioritize test questions, fall back to dataset questions
    display_questions = []
    
    if test_questions:
        display_questions = test_questions.copy()
    elif documents:
        # Fallback: get questions from dataset
        dataset_questions = []
        for doc in documents:
            q = doc.metadata.get('question')
            if q:
                dataset_questions.append(q)
        display_questions = dataset_questions[:CONFIG["MAX_SAMPLE_QUESTIONS"]]
    
    if display_questions:
        display_sample_questions(display_questions, question_mapping)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_input(
            "Enter your question:",
            value=st.session_state.current_question,
            placeholder="e.g., Should I use RAG or fine-tune my language model?",
            help="Ask anything related to AI engineering",
            key="question_input"
        )
        
        if question != st.session_state.current_question:
            st.session_state.current_question = question
    
    with col2:
        if st.button("Random Question") and display_questions:
            import random
            random_question = random.choice(display_questions)
            st.session_state.current_question = random_question
            st.rerun()
        
        if st.button("Clear"):
            st.session_state.current_question = ""
            st.rerun()
    
    current_question = st.session_state.current_question
    
    if current_question and st.button("Search", type="primary"):
        process_single_question(systems, selected_methods, current_question, top_k)


def batch_comparison_mode(systems: Dict[str, RAGSystem], top_k: int) -> None:
    """Batch comparison interface."""
    st.header("📊 Batch Question Comparison")
    st.write("Test multiple questions at once and see aggregate performance!")
    
    _, test_questions, _, _ = load_datasets()
    
    if not test_questions:
        st.error("No test questions available!")
        return
    
    st.subheader("📝 Select Questions to Test")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_questions = st.multiselect(
            "Choose questions:",
            test_questions,
            default=test_questions[:5],
            help="Select which questions to test"
        )
    
    with col2:
        if st.button("🎯 Select All"):
            st.session_state.selected_questions = test_questions
            st.rerun()
        
        if st.button("🎲 Random 5"):
            import random
            st.session_state.selected_questions = random.sample(test_questions, min(5, len(test_questions)))
            st.rerun()
    
    if selected_questions and st.button("🚀 Run Batch Test", type="primary"):
        run_batch_test(systems, selected_questions, top_k)


def run_batch_test(systems: Dict[str, RAGSystem], selected_questions: List[str], top_k: int) -> None:
    """Run batch testing across multiple questions and methods."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_results = {}
    
    for method_name, system in systems.items():
        status_text.text(f"Testing {method_name}...")
        method_results = []
        
        for i, question in enumerate(selected_questions):
            result = process_rag_query(system, question, top_k)
            method_results.append(result)
            progress_bar.progress((len(all_results) + (i + 1) / len(selected_questions)) / len(systems))
        
        all_results[method_name] = method_results
    
    progress_bar.empty()
    status_text.empty()
    
    display_batch_results(selected_questions, all_results)


def display_batch_results(questions: List[str], all_results: Dict[str, List[Dict]]) -> None:
    """Display batch comparison results."""
    st.subheader("📈 Batch Results Summary")
    
    summary_stats = {}
    for method, results in all_results.items():
        try:
            summary_stats[method] = RAGMetrics.calculate_summary_stats(results)
        except Exception as e:
            st.warning(f"Could not calculate stats for {method}: {e}")
            summary_stats[method] = {
                'success_rate': 0,
                'avg_response_time': 0,
                'avg_similarity': 0,
                'total_docs_retrieved': 0
            }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Questions Tested", len(questions))
    
    with col2:
        if summary_stats:
            best_method = max(summary_stats.keys(), key=lambda x: summary_stats[x]['success_rate'])
            st.metric("Best Method", best_method)
    
    with col3:
        if summary_stats:
            fastest_method = min(summary_stats.keys(), key=lambda x: summary_stats[x]['avg_response_time'])
            st.metric("Fastest Method", fastest_method)
    
    with col4:
        if summary_stats:
            most_accurate = max(summary_stats.keys(), key=lambda x: summary_stats[x]['avg_similarity'])
            st.metric("Most Accurate", most_accurate)
    
    if len(summary_stats) > 1:
        try:
            fig = create_comparison_chart(summary_stats)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create comparison chart: {e}")
    
    st.subheader("📊 Detailed Metrics")
    
    if summary_stats:
        df_data = []
        for method, stats in summary_stats.items():
            df_data.append({
                "Method": method,
                "Success Rate": f"{stats.get('success_rate', 0):.1%}",
                "Avg Response Time": f"{stats.get('avg_response_time', 0):.2f}s",
                "Avg Similarity": f"{stats.get('avg_similarity', 0):.3f}",
                "Total Docs Retrieved": stats.get('total_docs_retrieved', 0)
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
    
    with st.expander("🔍 Question-by-Question Breakdown"):
        for i, question in enumerate(questions):
            st.write(f"**Q{i+1}: {question}**")
            
            cols = st.columns(len(all_results))
            for j, (method, results) in enumerate(all_results.items()):
                with cols[j]:
                    if i < len(results):
                        result = results[i]
                        st.write(f"*{method}*")
                        st.write(f"📊 {result.get('avg_similarity', 0):.3f}")
                        st.write(f"⏱️ {result.get('retrieval_time', 0):.2f}s")


def full_experiment_mode(top_k: int) -> None:
    """Full experiment interface."""
    st.header("🧪 Full Experiment Mode")
    st.write("Run a comprehensive experiment and save results for analysis!")
    
    if st.button("🚀 Run Full Experiment", type="primary"):
        with st.spinner("Running comprehensive experiment..."):
            try:
                runner = ExperimentRunner()
                results = runner.run_single_experiment(
                    "ai_engineering_qa.json",
                    "test_questions.json",
                    top_k
                )
                
                if results:
                    filepath = runner.save_results(results)
                    st.success(f"✅ Experiment completed! Results saved to: {filepath}")
                    
                    summary_stats = results.get('summary_stats', {})
                    
                    if summary_stats:
                        st.subheader("📈 Experiment Results")
                        
                        try:
                            fig = create_comparison_chart(summary_stats)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not create visualization: {e}")
                        
                        for method, stats in summary_stats.items():
                            with st.container():
                                st.write(f"**{method.upper()}**")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Success Rate", f"{stats.get('success_rate', 0):.1%}")
                                with col2:
                                    st.metric("Avg Time", f"{stats.get('avg_response_time', 0):.2f}s")
                                with col3:
                                    st.metric("Avg Similarity", f"{stats.get('avg_similarity', 0):.3f}")
                                with col4:
                                    st.metric("Total Time", f"{stats.get('total_experiment_time', 0):.1f}s")
                        
                        try:
                            with open(filepath, 'r') as f:
                                file_content = f.read()
                            
                            st.download_button(
                                "📥 Download Results (JSON)",
                                data=file_content,
                                file_name="rag_experiment_results.json",
                                mime="application/json"
                            )
                        except Exception as e:
                            st.warning(f"Could not create download button: {e}")
                else:
                    st.error("❌ Experiment returned no results")
                    
            except Exception as e:
                st.error(f"❌ Experiment failed: {e}")


def add_sidebar_info() -> None:
    """Add comprehensive information to sidebar with explainability."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Model Information")
    
    st.sidebar.write("**Current Models:**")
    st.sidebar.write("🔧 **TF-IDF**: scikit-learn TfidfVectorizer")
    st.sidebar.write("   • Max features: 1,000")
    st.sidebar.write("   • N-grams: 1-2 (words + phrases)")
    st.sidebar.write("   • Stop words: English")
    
    st.sidebar.write("🤗 **Embeddings**: all-MiniLM-L6-v2")  
    st.sidebar.write("   • Dimensions: 384")
    st.sidebar.write("   • Model size: ~23MB")
    st.sidebar.write("   • Training: 1B+ sentence pairs")
    
    st.sidebar.write("🤖 **LLM**: Llama 3.2 (via Ollama)")
    st.sidebar.write("   • Temperature: 0.3")
    st.sidebar.write("   • Context: Up to 2K tokens")
    
    try:
        documents, test_questions, _ = load_datasets()
        if documents:
            st.sidebar.markdown("---")
            st.sidebar.subheader("📚 Dataset Info")
            st.sidebar.write(f"**Q&A Pairs**: {len(documents)}")
            st.sidebar.write(f"**Test Questions**: {len(test_questions) if test_questions else 0}")
            
            categories = set()
            for doc in documents:
                cat = doc.metadata.get('category', 'general')
                categories.add(cat)
            
            st.sidebar.write(f"**Categories**: {', '.join(sorted(categories))}")
    except Exception:
        st.sidebar.write("📊 Dataset: Loading...")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Parameters Explained")
    st.sidebar.write("**Top-K**: Retrieve most similar documents")
    st.sidebar.write("**Similarity Threshold**: 0.1 (minimum relevance)")
    st.sidebar.write("**Cosine Similarity**: Measures angle between vectors (0-1)")
    st.sidebar.write("**Generation Quality**: Cosine similarity between expected and generated answers")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Expected Performance")
    st.sidebar.write("**TF-IDF**:")
    st.sidebar.write("   • Speed: ~0.1s")
    st.sidebar.write("   • Good for: Exact keyword matches")
    st.sidebar.write("   • Struggles with: Synonyms, context")
    
    st.sidebar.write("**Embeddings**:")
    st.sidebar.write("   • Speed: ~0.2s")  
    st.sidebar.write("   • Good for: Semantic meaning")
    st.sidebar.write("   • Struggles with: Very specific terms")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Learn More")
    st.sidebar.write("""
    - [RAG Paper](https://arxiv.org/abs/2005.11401)
    - [Sentence Transformers](https://www.sbert.net/)
    - [LangChain Docs](https://python.langchain.com/)
    """)


def main() -> None:
    """Main Streamlit application."""
    st.markdown(f"""
    <div class="main-header">
        <h1>{CONFIG["PAGE_ICON"]} {CONFIG["PAGE_TITLE"]}</h1>
        <p>Interactive RAG System Comparison</p>
    </div>
    """, unsafe_allow_html=True)
    
    systems, error = initialize_rag_systems()
    
    if error:
        st.error(f"❌ {error}")
        st.info("💡 Make sure you have the data files. Run: `python -c 'from src.datasets import setup_sample_data; setup_sample_data()'`")
        return
    
    st.sidebar.title("🛠️ Configuration")
    
    mode = st.sidebar.selectbox(
        "Choose Mode",
        ["🔍 Single Question", "📊 Batch Comparison", "🧪 Full Experiment"],
        help="Select how you want to explore RAG systems"
    )
    
    selected_methods = []
    if mode == "🔍 Single Question":
        selected_methods = st.sidebar.multiselect(
            "Select Methods to Compare",
            list(systems.keys()),
            default=list(systems.keys()),
            help="Choose which RAG methods to test"
        )
    
    st.sidebar.subheader("🎛️ Parameters")
    top_k = st.sidebar.slider(
        "Top-K Documents", 
        1, 5, CONFIG["DEFAULT_TOP_K"], 
        help="Number of documents to retrieve"
    )
    
    try:
        if mode == "🔍 Single Question":
            single_question_mode(systems, selected_methods, top_k)
        elif mode == "📊 Batch Comparison":
            batch_comparison_mode(systems, top_k)
        else:
            full_experiment_mode(top_k)
    except Exception as e:
        st.error(f"❌ Error in {mode}: {e}")
        st.write("Please try refreshing the page or contact support if the issue persists.")


if __name__ == "__main__":
    add_sidebar_info()
    main()