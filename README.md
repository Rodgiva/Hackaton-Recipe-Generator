# AI Recipe Generator

Generates an original recipe from a list of ingredients, using RAG over 100 000 real recipes.

Requires Python 3.12.

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

### 2. Install PyTorch

With an NVIDIA GPU (CUDA 12.6):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Without a GPU:

```bash
pip install torch
```

### 3. Install the dependencies

```bash
pip install transformers==4.46.0 accelerate==1.15.0 datasets==2.21.0 sentence-transformers==5.7.0 langchain-community==0.3.31 langchain-core==0.3.86 langchain-huggingface==0.1.2 faiss-cpu==1.15.1 nltk==3.10.3 streamlit==1.66.1
```

### 4. Download the WordNet corpus

```bash
python -m nltk.downloader wordnet
```

### 5. Build the FAISS index

```bash
python build_index.py
```

## Run

```bash
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`.

## Use

1. Enter your ingredients, separated by commas: `eggs, potatoes, onions, cheese`.
2. Click **Generate recipe**.
3. Read the recipe: title, servings, ingredients with quantities, numbered steps.

Expand *Real recipes used as RAG context* to see the three dataset recipes used as context, and *Prompt sent to the model* to see the full prompt.
