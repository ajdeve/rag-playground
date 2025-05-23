"""
Interactive Streamlit app for RAG system demonstration and experimentation.
Perfect for Medium article screenshots and user interaction.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
from typing import Dict, List, Any

# Import our RAG components
try:
    from src.rag_core import RAGSystem, TFIDFRetriever, EmbeddingRetriever, LLMInterface
    from src.datasets import DatasetLoader
    from src.evaluation import ExperimentRunner, RAGMetrics, GenerationEvaluator
except ImportError:
    st.error("❌ Cannot import RAG modules. Make sure you're running from the project root directory.")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="RAG Learning Lab",
    page_icon="🤖",
    layout="wide",
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
def load_datasets():
    """Load datasets with caching for performance"""
    try:
        loader = DatasetLoader()
        documents = loader.load_qa_dataset("ai_engineering_qa.json")
        test_questions = loader.load_test_questions("test_questions.json")
        return documents, test_questions, None
    except FileNotFoundError as e:
        return None, None, str(e)


@st.cache_resource
def initialize_rag_systems():
    """Initialize RAG systems with caching"""
    documents, _, error = load_datasets()
    
    if error or not documents:
        return None, error
    
    try:
        # Create both systems
        llm = LLMInterface()
        systems = {
            "TF-IDF": RAGSystem(TFIDFRetriever(), llm),
            "Embeddings": RAGSystem(EmbeddingRetriever(), llm)
        }
        
        # Load documents into both systems
        for system in systems.values():
            system.load_documents(documents)
        
        return systems, None
        
    except Exception as e:
        return None, f"Error initializing RAG systems: {e}"


def create_comparison_chart(results: Dict[str, Dict]) -> go.Figure:
    """Create a comparison chart for RAG methods"""
    methods = list(results.keys())
    metrics = ['success_rate', 'avg_response_time', 'avg_similarity']
    
    fig = go.Figure()
    
    for method in methods:
        values = [
            results[method].get('success_rate', 0) * 100,  # Convert to percentage
            results[method].get('avg_response_time', 0),
            results[method].get('avg_similarity', 0) * 100  # Convert to percentage
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


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG Learning Lab</h1>
        <p>Interactive RAG System Comparison</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize systems
    systems, error = initialize_rag_systems()
    
    if error:
        st.error(f"❌ {error}")
        st.info("💡 Make sure you have the data files. Run: `python -c 'from src.datasets import setup_sample_data; setup_sample_data()'`")
        return
    
    # Sidebar configuration
    st.sidebar.title("🛠️ Configuration")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Choose Mode",
        ["🔍 Single Question", "📊 Batch Comparison", "🧪 Full Experiment"],
        help="Select how you want to explore RAG systems"
    )
    
    # Method selection for single question mode
    if mode == "🔍 Single Question":
        selected_methods = st.sidebar.multiselect(
            "Select Methods to Compare",
            list(systems.keys()),
            default=list(systems.keys()),
            help="Choose which RAG methods to test"
        )
    
    # Retrieval parameters
    st.sidebar.subheader("🎛️ Parameters")
    top_k = st.sidebar.slider("Top-K Documents", 1, 5, 3, help="Number of documents to retrieve")
    
    # Main content based on mode
    if mode == "🔍 Single Question":
        single_question_mode(systems, selected_methods, top_k)
    elif mode == "📊 Batch Comparison":
        batch_comparison_mode(systems, top_k)
    else:
        full_experiment_mode(top_k)


def single_question_mode(systems: Dict, selected_methods: List[str], top_k: int):
    """Single question comparison interface"""
    
    st.header("Single Question Analysis")
    st.write("Ask a question and see how different RAG methods respond!")
    
    # Load sample questions for quick testing
    _, test_questions, _ = load_datasets()
    
    # Initialize session state for question if it doesn't exist
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    
    # Sample questions with clickable buttons
    if test_questions:
        st.write("**Click a sample question to try:**")
        sample_cols = st.columns(3)
        for i, sample_q in enumerate(test_questions[:6]):  # Show first 6 questions
            with sample_cols[i % 3]:
                if st.button(f"{sample_q[:35]}...", key=f"sample_{i}", help=sample_q):
                    st.session_state.current_question = sample_q
                    st.rerun()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Use session state value as the default
        question = st.text_input(
            "Enter your question:",
            value=st.session_state.current_question,
            placeholder="e.g., What is the difference between RAG and fine-tuning?",
            help="Ask anything related to AI engineering",
            key="question_input"
        )
        
        # Update session state when text input changes
        if question != st.session_state.current_question:
            st.session_state.current_question = question
    
    with col2:
        if st.button("Random Question") and test_questions:
            import random
            random_question = random.choice(test_questions)
            st.session_state.current_question = random_question
            st.rerun()
        
        # Clear button
        if st.button("Clear"):
            st.session_state.current_question = ""
            st.rerun()
    
    # Use the session state question for processing
    current_question = st.session_state.current_question
    
    if current_question and st.button("Search", type="primary"):
        
        # Progress bar for user experience
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize generation evaluator for answer quality assessment
        gen_evaluator = GenerationEvaluator()
        
        results = {}
        
        for i, method_name in enumerate(selected_methods):
            status_text.text(f"Processing with {method_name}...")
            progress_bar.progress((i + 1) / len(selected_methods))
            
            system = systems[method_name]
            result = system.ask(current_question, top_k)
            
            # Try to find expected answer for generation quality evaluation
            documents, _, _ = load_datasets()
            expected_answer = None
            
            if documents:
                # Simple matching logic
                for doc in documents:
                    doc_question = doc.metadata.get('question', '')
                    if current_question.lower() in doc_question.lower() or doc_question.lower() in current_question.lower():
                        expected_answer = doc.metadata.get('expected_answer')
                        break
            
            # Add generation quality if we found an expected answer
            if expected_answer:
                answer_similarity = gen_evaluator.evaluate_answer_similarity(
                    result.get('answer', ''), expected_answer
                )
                result['expected_answer'] = expected_answer
                result['answer_similarity'] = answer_similarity
            
            results[method_name] = result
        
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        display_single_question_results(current_question, results)


def display_single_question_results(question: str, results: Dict):
    """Display results for single question comparison with full explainability"""
    
    st.subheader(f"📋 Results for: *{question}*")
    
    # Load documents for detailed display
    documents, _, _ = load_datasets()
    
    # Enhanced metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Methods Tested", len(results))
    
    with col2:
        avg_time = sum(r.get('retrieval_time', 0) for r in results.values()) / len(results)
        st.metric("Avg Response Time", f"{avg_time:.2f}s")
    
    with col3:
        successful = sum(1 for r in results.values() if r.get('docs_found', 0) > 0)
        st.metric("Successful Retrievals", f"{successful}/{len(results)}")
    
    with col4:
        similarities = [r.get('avg_similarity', 0) for r in results.values() if r.get('docs_found', 0) > 0]
        avg_sim = sum(similarities) / len(similarities) if similarities else 0
        st.metric("Avg Similarity", f"{avg_sim:.3f}")
    
    # Add explainability section
    st.subheader("Understanding the Comparison")
    
    # Quick insights
    if len(results) == 2:
        methods = list(results.keys())
        method1, method2 = methods[0], methods[1]
        result1, result2 = results[method1], results[method2]
        
        # Compare performance
        better_similarity = method1 if result1.get('avg_similarity', 0) > result2.get('avg_similarity', 0) else method2
        faster_method = method1 if result1.get('retrieval_time', 0) < result2.get('retrieval_time', 0) else method2
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Better Retrieval:** {better_similarity} found more relevant content")
        with col2:
            st.info(f"**Faster Method:** {faster_method} responded quicker")
        
        # Show if they retrieved different documents
        docs1 = set(result1.get('retrieved_qa_ids', []))
        docs2 = set(result2.get('retrieved_qa_ids', []))
        
        if docs1 != docs2:
            st.warning(f"**Different Documents Retrieved:** Methods found different content sources")
            unique_to_1 = docs1 - docs2
            unique_to_2 = docs2 - docs1
            
            if unique_to_1:
                st.write(f"Only {method1} found: {', '.join(unique_to_1)}")
            if unique_to_2:
                st.write(f"Only {method2} found: {', '.join(unique_to_2)}")
        else:
            st.success("**Same Documents:** Both methods retrieved identical sources")
    
    # Side-by-side comparison with enhanced details
    if len(results) == 2:
        col1, col2 = st.columns(2)
        methods = list(results.keys())
        
        for i, (method, result) in enumerate(results.items()):
            with col1 if i == 0 else col2:
                display_method_result(method, result, documents)
    else:
        # Stacked display for multiple methods
        for method, result in results.items():
            display_method_result(method, result, documents)


def display_method_result(method: str, result: Dict, documents: List = None):
    """Display result for a single method with full explainability"""
    
    # Get model info for embedding methods
    model_info = ""
    if "embedding" in method.lower():
        model_info = " (all-MiniLM-L6-v2)"
    
    st.markdown(f"""
    <div class="comparison-box">
        <h4>{method}{model_info}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced metrics with model info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Time", f"{result.get('retrieval_time', 0):.2f}s")
    with col2:
        st.metric("Docs Found", result.get('docs_found', 0))
    with col3:
        st.metric("Avg Similarity", f"{result.get('avg_similarity', 0):.3f}")
    with col4:
        st.metric("Context Length", f"{result.get('context_length', 0)} chars")
    
    # Method-specific explainability - only for TF-IDF
    if "tfidf" in method.lower():
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
    
    # Answer with confidence indicator
    answer = result.get('answer', 'No answer generated')
    confidence = "High" if result.get('avg_similarity', 0) > 0.7 else "Medium" if result.get('avg_similarity', 0) > 0.4 else "Low"
    confidence_indicator = "●" if confidence == "High" else "◐" if confidence == "Medium" else "○"
    
    st.write(f"**Answer ({confidence_indicator} {confidence} Confidence):**")
    st.write(answer)
    
    # Show retrieved document sources clearly
    if result.get('docs_found', 0) > 0:
        # Get QA IDs from the result, with fallback
        qa_ids = result.get('retrieved_qa_ids', [])
        
        # Quick source summary
        if qa_ids:
            sources_text = ', '.join(qa_ids)
            st.write(f"**Sources Found**: `{sources_text}`")
        else:
            st.write(f"**Sources Found**: {result.get('docs_found', 0)} documents (IDs not available)")
        
        # Detailed retrieved documents
        with st.expander("View Retrieved Documents & Similarity Scores", expanded=False):
            scores = result.get('similarity_scores', [])
            
            if not documents:
                st.warning("Document details not available - documents list is empty")
                for i, score in enumerate(scores, 1):
                    qa_id = qa_ids[i-1] if i-1 < len(qa_ids) else f"unknown_{i}"
                    st.write(f"**Document {i}**: {qa_id} (Similarity: {score:.3f})")
                return
            
            for i, score in enumerate(scores, 1):
                qa_id = qa_ids[i-1] if i-1 < len(qa_ids) else f"document_{i}"
                
                # Find the actual document content
                doc_content = "Content not available"
                doc_question = "Question not found"
                doc_category = "Unknown"
                doc_tags = []
                
                # Try to find matching document
                if qa_id != f"document_{i}":  # We have a real QA ID
                    for doc in documents:
                        if doc.metadata.get('qa_id') == qa_id:
                            doc_content = doc.page_content
                            doc_question = doc.metadata.get('question', 'Unknown question')
                            doc_category = doc.metadata.get('category', 'general')
                            doc_tags = doc.metadata.get('tags', [])
                            break
                else:
                    # Fallback: show info for document at this index if available
                    if i-1 < len(documents):
                        doc = documents[i-1]
                        doc_content = doc.page_content
                        doc_question = doc.metadata.get('question', 'Unknown question')
                        doc_category = doc.metadata.get('category', 'general')
                        doc_tags = doc.metadata.get('tags', [])
                        qa_id = doc.metadata.get('qa_id', f"document_{i}")
                
                # Simple relevance indicator
                if score > 0.7:
                    relevance_indicator = "●"
                    score_label = "High Relevance"
                elif score > 0.4:
                    relevance_indicator = "◐"
                    score_label = "Medium Relevance"
                else:
                    relevance_indicator = "○"
                    score_label = "Low Relevance"
                
                st.markdown(f"""
                ### Document {i}: `{qa_id}`
                **{relevance_indicator} Similarity Score: {score:.3f}** ({score_label})
                - **Category:** `{doc_category}`
                - **Tags:** {', '.join(f'`{tag}`' for tag in doc_tags) if doc_tags else 'None'}
                - **Original Question:** *{doc_question}*
                """)
                
                # Show the actual content that was retrieved
                st.markdown("**Retrieved Content:**")
                st.code(doc_content, language="text")
                
                # Explain why this document was retrieved - simplified
                if "tfidf" in method.lower():
                    st.info(f"**Why {method} selected this**: Shares important keywords with your query")
                else:
                    st.info(f"**Why {method} selected this**: Semantically similar meaning to your query")
                
                if i < len(scores):
                    st.markdown("---")
            
            # Summary of retrieval quality
            st.markdown("### Retrieval Quality Summary")
            avg_score = sum(scores) / len(scores) if scores else 0
            
            # Debug the calculations step by step
            st.write("**Debug Step-by-Step Calculation:**")
            st.write(f"1. Similarity scores: {scores}")
            st.write(f"2. QA IDs: {qa_ids}")
            st.write(f"3. Documents available: {len(documents) if documents else 0}")
            
            # Calculate high quality docs with debugging
            high_quality_scores = [s for s in scores if s > 0.7]
            high_quality = len(high_quality_scores)
            st.write(f"4. Scores > 0.7: {high_quality_scores}")
            st.write(f"5. High quality count: {high_quality}")
            
            # Calculate context chars with detailed debugging
            total_chars = 0
            matched_docs = []
            
            if documents and qa_ids:
                st.write(f"6. Looking for documents with QA IDs in: {qa_ids}")
                for doc in documents:
                    doc_qa_id = doc.metadata.get('qa_id')
                    st.write(f"   - Checking document: {doc_qa_id}")
                    if doc_qa_id in qa_ids:
                        char_count = len(doc.page_content)
                        total_chars += char_count
                        matched_docs.append(f"{doc_qa_id}({char_count} chars)")
                        st.write(f"     ✓ MATCH! Added {char_count} chars")
                    else:
                        st.write(f"     ✗ No match")
                
                st.write(f"7. Matched documents: {matched_docs}")
                st.write(f"8. Total chars calculated: {total_chars}")
            else:
                st.write(f"6. PROBLEM: documents={documents is not None}, qa_ids={qa_ids}")
                # Use fallback
                total_chars = result.get('context_length', 0)
                st.write(f"7. Using fallback context_length: {total_chars}")
            
            # Show the metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Score", f"{avg_score:.3f}")
            with col2:
                st.metric("High Quality Docs", f"{high_quality}/{len(scores)}")
            with col3:
                st.metric("Total Context", f"{total_chars} chars")
            
            # Also show what we got from the RAG result directly
            st.write("**Direct from RAG Result:**")
            st.write(f"- context_length: {result.get('context_length', 'NOT FOUND')}")
            st.write(f"- avg_similarity: {result.get('avg_similarity', 'NOT FOUND')}")
            st.write(f"- docs_found: {result.get('docs_found', 'NOT FOUND')}")
            st.write(f"- All result keys: {list(result.keys())}")
                
            # Quick test: let's manually check the first document
            if documents:
                st.write("**Manual Document Check:**")
                first_doc = documents[0]
                st.write(f"- First document QA ID: {first_doc.metadata.get('qa_id', 'NO QA_ID')}")
                st.write(f"- First document content length: {len(first_doc.page_content)}")
                st.write(f"- First document metadata: {first_doc.metadata}")
                
                # Check if any retrieved QA ID matches first document
                if qa_ids and first_doc.metadata.get('qa_id') in qa_ids:
                    st.write("✓ First document IS in retrieved list")
                else:
                    st.write("✗ First document NOT in retrieved list")
    else:
        st.warning("No relevant documents found - try rephrasing your question")


def batch_comparison_mode(systems: Dict, top_k: int):
    """Batch comparison interface"""
    
    st.header("📊 Batch Question Comparison")
    st.write("Test multiple questions at once and see aggregate performance!")
    
    # Load test questions
    _, test_questions, _ = load_datasets()
    
    if not test_questions:
        st.error("No test questions available!")
        return
    
    # Question selection
    st.subheader("📝 Select Questions to Test")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_questions = st.multiselect(
            "Choose questions:",
            test_questions,
            default=test_questions[:5],  # Default to first 5
            help="Select which questions to test"
        )
    
    with col2:
        if st.button("🎯 Select All"):
            selected_questions = test_questions
            st.rerun()
        
        if st.button("🎲 Random 5"):
            import random
            selected_questions = random.sample(test_questions, min(5, len(test_questions)))
            st.rerun()
    
    if selected_questions and st.button("🚀 Run Batch Test", type="primary"):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = {}
        
        # Test each method
        for method_name, system in systems.items():
            status_text.text(f"Testing {method_name}...")
            method_results = []
            
            for i, question in enumerate(selected_questions):
                result = system.ask(question, top_k)
                method_results.append(result)
                progress_bar.progress((len(all_results) + (i + 1) / len(selected_questions)) / len(systems))
            
            all_results[method_name] = method_results
        
        progress_bar.empty()
        status_text.empty()
        
        # Display batch results
        display_batch_results(selected_questions, all_results)


def display_batch_results(questions: List[str], all_results: Dict):
    """Display batch comparison results"""
    
    st.subheader("📈 Batch Results Summary")
    
    # Calculate summary statistics
    summary_stats = {}
    for method, results in all_results.items():
        summary_stats[method] = RAGMetrics.calculate_summary_stats(results)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Questions Tested", len(questions))
    
    with col2:
        best_method = max(summary_stats.keys(), key=lambda x: summary_stats[x]['success_rate'])
        st.metric("Best Method", best_method)
    
    with col3:
        fastest_method = min(summary_stats.keys(), key=lambda x: summary_stats[x]['avg_response_time'])
        st.metric("Fastest Method", fastest_method)
    
    with col4:
        most_accurate = max(summary_stats.keys(), key=lambda x: summary_stats[x]['avg_similarity'])
        st.metric("Most Accurate", most_accurate)
    
    # Comparison chart
    if len(summary_stats) > 1:
        fig = create_comparison_chart(summary_stats)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.subheader("📊 Detailed Metrics")
    
    df_data = []
    for method, stats in summary_stats.items():
        df_data.append({
            "Method": method,
            "Success Rate": f"{stats['success_rate']:.1%}",
            "Avg Response Time": f"{stats['avg_response_time']:.2f}s",
            "Avg Similarity": f"{stats['avg_similarity']:.3f}",
            "Total Docs Retrieved": stats['total_docs_retrieved']
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
    
    # Question-by-question breakdown
    with st.expander("🔍 Question-by-Question Breakdown"):
        for i, question in enumerate(questions):
            st.write(f"**Q{i+1}: {question}**")
            
            cols = st.columns(len(all_results))
            for j, (method, results) in enumerate(all_results.items()):
                with cols[j]:
                    result = results[i]
                    st.write(f"*{method}*")
                    st.write(f"📊 {result.get('avg_similarity', 0):.3f}")
                    st.write(f"⏱️ {result.get('retrieval_time', 0):.2f}s")


def full_experiment_mode(top_k: int):
    """Full experiment interface"""
    
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
                    # Save results
                    filepath = runner.save_results(results)
                    
                    st.success(f"✅ Experiment completed! Results saved to: {filepath}")
                    
                    # Display summary
                    summary_stats = results.get('summary_stats', {})
                    
                    if summary_stats:
                        st.subheader("📈 Experiment Results")
                        
                        # Create comparison visualization
                        fig = create_comparison_chart(summary_stats)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show detailed metrics
                        for method, stats in summary_stats.items():
                            with st.container():
                                st.write(f"**{method.upper()}**")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Success Rate", f"{stats['success_rate']:.1%}")
                                with col2:
                                    st.metric("Avg Time", f"{stats['avg_response_time']:.2f}s")
                                with col3:
                                    st.metric("Avg Similarity", f"{stats['avg_similarity']:.3f}")
                                with col4:
                                    st.metric("Total Time", f"{stats['total_experiment_time']:.1f}s")
                    
                    # Download results
                    st.download_button(
                        "📥 Download Results (JSON)",
                        data=open(filepath, 'r').read(),
                        file_name=f"rag_experiment_results.json",
                        mime="application/json"
                    )
                
            except Exception as e:
                st.error(f"❌ Experiment failed: {e}")


# Sidebar info
    # Add explainability to sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Model Information")
    
    # Show current models being used
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
    
    # Add dataset info
    documents, test_questions, _ = load_datasets()
    if documents:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📚 Dataset Info")
        st.sidebar.write(f"**Q&A Pairs**: {len(documents)}")
        st.sidebar.write(f"**Test Questions**: {len(test_questions) if test_questions else 0}")
        
        # Show categories
        categories = set()
        for doc in documents:
            cat = doc.metadata.get('category', 'general')
            categories.add(cat)
        
        st.sidebar.write(f"**Categories**: {', '.join(sorted(categories))}")
    
    # Similarity threshold explanation
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Parameters Explained")
    st.sidebar.write(f"**Top-K**: Retrieve {top_k} most similar documents")
    st.sidebar.write("**Similarity Threshold**: 0.1 (minimum relevance)")
    st.sidebar.write("**Cosine Similarity**: Measures angle between vectors (0-1)")
    
    # Performance expectations
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


def add_sidebar_info():
    """Add comprehensive information to sidebar with explainability"""
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Model Information")
    
    # Show current models being used
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
    
    # Add dataset info
    try:
        documents, test_questions, _ = load_datasets()
        if documents:
            st.sidebar.markdown("---")
            st.sidebar.subheader("📚 Dataset Info")
            st.sidebar.write(f"**Q&A Pairs**: {len(documents)}")
            st.sidebar.write(f"**Test Questions**: {len(test_questions) if test_questions else 0}")
            
            # Show categories
            categories = set()
            for doc in documents:
                cat = doc.metadata.get('category', 'general')
                categories.add(cat)
            
            st.sidebar.write(f"**Categories**: {', '.join(sorted(categories))}")
    except:
        st.sidebar.write("📊 Dataset: Loading...")
    
    # Similarity threshold explanation
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Parameters Explained")
    st.sidebar.write("**Top-K**: Retrieve most similar documents")
    st.sidebar.write("**Similarity Threshold**: 0.1 (minimum relevance)")
    st.sidebar.write("**Cosine Similarity**: Measures angle between vectors (0-1)")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Learn More")
    st.sidebar.write("""
    - [RAG Paper](https://arxiv.org/abs/2005.11401)
    - [Sentence Transformers](https://www.sbert.net/)
    - [LangChain Docs](https://python.langchain.com/)
    """)


if __name__ == "__main__":
    add_sidebar_info()
    main()