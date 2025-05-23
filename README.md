# RAG Learning Lab

Interactive Streamlit app for comparing RAG system approaches.

## Quick Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install and start Ollama:**
   ```bash
   # Install Ollama (see https://ollama.ai)
   ollama pull llama3.2
   ```

3. **Run the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

## Features

- 🔧 **TF-IDF vs Embeddings**: Compare different retrieval methods
- 📊 **Generation Quality**: Evaluate answer accuracy against expected responses  
- 🎯 **Interactive UI**: Clean interface with real-time comparisons
- 📈 **Batch Testing**: Test multiple questions at once

## Sample Questions

Try these questions to see generation quality metrics:

- "Should I use RAG or fine-tune my language model?"
- "How much do OpenAI embeddings cost compared to local models?"
- "Why is my RAG system making things up?"

## Project Structure

```
├── streamlit_app.py          # Main Streamlit application
├── src/
│   ├── rag_core.py          # RAG system implementation
│   ├── datasets.py          # Dataset loading utilities
│   └── evaluation.py       # Evaluation and metrics
├── data/
│   ├── ai_engineering_qa.json    # Q&A dataset
│   └── test_questions.json       # Test questions
└── results/                 # Experiment results
```

## Troubleshooting

- **Import errors**: Make sure you're running from the project root
- **Ollama errors**: Ensure Ollama is running and llama3.2 is installed
- **No generation quality**: Questions must match dataset entries exactly

Happy experimenting! 🚀
