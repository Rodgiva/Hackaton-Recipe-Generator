import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import torch
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from nltk.stem import WordNetLemmatizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_NAME = "LIUH0115/RecipeNLG_Group8"
SAMPLE_SIZE = 100_000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FETCH_K = 30
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

CUDA = torch.cuda.is_available()
LLM_DEVICE = 0 if CUDA else -1
LLM_DTYPE = torch.float16 if CUDA else "auto"
EMBED_DEVICE = "cuda" if CUDA else "cpu"


# ---------------------------------------------------------------------------
# 1) Data loading
# ---------------------------------------------------------------------------
def load_recipe_sample(
    n: int = SAMPLE_SIZE,
    seed: int = 42,
    exclude_ids: Optional[Set[int]] = None,
) -> List[dict]:
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    exclude_ids = exclude_ids or set()

    rows = []
    for row in ds:
        if row.get("id") in exclude_ids:
            continue
        ingredients = row.get("normalized_ingredients") or []
        directions = row.get("directions") or []
        if 2 <= len(ingredients) <= 12 and len(directions) >= 1:
            rows.append(row)
        if len(rows) >= n:
            break
    return rows


# ---------------------------------------------------------------------------
# 2) Documents
# ---------------------------------------------------------------------------
def build_documents(rows: List[dict]) -> List[Document]:
    documents = []
    for row in rows:
        normalized = [
            ing.lower().strip() for ing in row.get("normalized_ingredients") or []
        ]
        normalized = [ing for ing in normalized if ing]
        documents.append(
            Document(
                page_content=", ".join(normalized),
                metadata={
                    "title": row.get("title", ""),
                    "normalized_ingredients": normalized,
                    "ingredient_lines": row.get("ingredients") or [],
                    "directions": row.get("directions") or [],
                    "link": row.get("link", ""),
                },
            )
        )
    return documents


# ---------------------------------------------------------------------------
# 3) Word matching (morphology only -- no food vocabulary involved)
# ---------------------------------------------------------------------------
_NORMALIZE_WORD = WordNetLemmatizer().lemmatize

def normalize_word(word: str) -> str:
    return _NORMALIZE_WORD(word.lower())

def lemma_tokens(text: str) -> Tuple[str, ...]:
    return tuple(normalize_word(w) for w in re.findall(r"[a-z]+", text.lower()))

def _token_span(term_tokens: Tuple[str, ...], text_tokens: Tuple[str, ...]) -> int:
    n = len(term_tokens)
    if not n or n > len(text_tokens):
        return -1
    for i in range(len(text_tokens) - n + 1):
        if text_tokens[i:i + n] == term_tokens:
            return i
    return -1


def _mentions(term: str, text_lower: str) -> bool:
    return _token_span(lemma_tokens(term), lemma_tokens(text_lower)) >= 0


# ---------------------------------------------------------------------------
# 4) Retrieval
# ---------------------------------------------------------------------------
def load_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs={"device": EMBED_DEVICE}
    )

def build_or_load_vectorstore(
    documents: List[Document],
    index_dir: str = INDEX_DIR,
    embedding_model: str = EMBEDDING_MODEL,
    force_rebuild: bool = False,
) -> FAISS:
    embeddings = load_embeddings(embedding_model)

    if not force_rebuild and os.path.isdir(index_dir):
        return FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True
        )

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(index_dir)
    return vectorstore


def get_retriever(vectorstore: FAISS, k: int = FETCH_K):
    return vectorstore.as_retriever(search_kwargs={"k": k})


def count_covered(doc: Document, ingredients: List[str]) -> int:
    listed = " , ".join(doc.metadata.get("normalized_ingredients", [])).lower()
    return sum(1 for ing in ingredients if _mentions(ing, listed))


def rerank_by_coverage(
    docs: List[Document], ingredients: List[str], top_n: int = 3
) -> List[Document]:
    ranked = sorted(docs, key=lambda d: count_covered(d, ingredients), reverse=True)
    return ranked[:top_n]


def build_context(docs: List[Document], max_docs: int = 3, max_chars: int = 300) -> str:
    chunks = []
    for doc in docs[:max_docs]:
        lines = doc.metadata.get("ingredient_lines") or doc.metadata.get(
            "normalized_ingredients", []
        )
        steps = " ".join(doc.metadata.get("directions", []))
        if len(steps) > max_chars:
            steps = steps[:max_chars].rsplit(" ", 1)[0] + " ..."
        chunks.append(
            "Example recipe:\n"
            + "\n".join(f"  {line}" for line in lines)
            + f"\n  Steps: {steps}"
        )
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# 5) Generation
# ---------------------------------------------------------------------------
LLM_SYSTEM = (
    "You are a precise cooking assistant. You follow ingredient constraints exactly."
)

LLM_PROMPT = """Write an original recipe that uses ONLY these ingredients: {ingredients}.
You may also use basic seasoning you would expect in any kitchen. Do not build the dish around anything else.

Here are real recipes with similar ingredients, for the quantities they use and their technique. They are written in US units: convert everything to metric.
{context}

Use METRIC units only:
- liquids in ml, or in L above 1000 ml
- everything else in g, or in kg above 1000 g
- countable items stay a count ("3 eggs", "2 onions"), never a weight
- round to readable values: 200 g not 187 g, 250 ml not 237 ml
- temperatures in degrees Celsius
Never write cups, tablespoons, teaspoons, ounces, pounds or Fahrenheit.

Reply in exactly this format, nothing else:
Title: <a short dish name>
Serves: <number of people>
Ingredients:
- <amount + ingredient, e.g. "500 g potatoes, diced">
- <amount + ingredient>
Steps:
1. <first action>
2. <second action>
3. ...
Give an amount for every ingredient. Write 4 to 7 steps, each one concrete cooking action with times or temperatures where useful."""

_STEP_NUMBER = r"(?:[1-9]|[12]\d)[.:)]"

def load_llm(model_name: str = LLM_MODEL):
    return pipeline(
        task="text-generation",
        model=model_name,
        device=LLM_DEVICE,
        torch_dtype=LLM_DTYPE,
        max_new_tokens=400,
        do_sample=False,
        return_full_text=False,
    )

_VOLUME_ML = {
    "tsp": 5, "teaspoon": 5, "teaspoons": 5,
    "tbsp": 15, "tablespoon": 15, "tablespoons": 15,
    "cup": 240, "cups": 240,
    "fl oz": 30, "fluid ounce": 30, "fluid ounces": 30,
    "pint": 473, "pints": 473, "quart": 946, "quarts": 946,
}
_WEIGHT_G = {
    "oz": 28, "ounce": 28, "ounces": 28,
    "lb": 450, "lbs": 450, "pound": 450, "pounds": 450,
}

_AMOUNT = r"(\d+\s+\d+/\d+|\d+/\d+|\d+(?:[.,]\d+)?)"

def _amount_value(raw: str) -> float:
    raw = raw.replace(",", ".").strip()
    if " " in raw:
        whole, frac = raw.split(None, 1)
        return float(whole) + _amount_value(frac)
    if "/" in raw:
        num, den = raw.split("/")
        return float(num) / float(den)
    return float(raw)


def _round_readable(value: float) -> int:
    step = 1 if value < 20 else 5 if value < 100 else 10
    return max(step, int(value / step + 0.5) * step)


def to_metric(text: str) -> str:
    def convert_measure(match):
        amount, unit = match.group(1), match.group(2).lower()
        try:
            value = _amount_value(amount)
        except (ValueError, ZeroDivisionError):
            return match.group(0)

        if unit in _VOLUME_ML:
            ml = value * _VOLUME_ML[unit]
            if ml >= 1000:
                return f"{round(ml / 1000, 1):g} L"
            return f"{_round_readable(ml)} ml"

        grams = value * _WEIGHT_G[unit]
        if grams >= 1000:
            return f"{round(grams / 1000, 1):g} kg"
        return f"{_round_readable(grams)} g"

    units = sorted(
        list(_VOLUME_ML) + list(_WEIGHT_G), key=len, reverse=True
    )
    pattern = rf"{_AMOUNT}\s*({'|'.join(re.escape(u) for u in units)})\b\.?"
    text = re.sub(pattern, convert_measure, text, flags=re.IGNORECASE)

    def convert_temp(match):
        celsius = (float(match.group(1)) - 32) * 5 / 9
        return f"{max(5, int(celsius / 5 + 0.5) * 5)}°C"

    text = re.sub(
        r"(\d{2,3})\s*(?:°\s*|degrees?\s+)?F(?:ahrenheit)?\b",
        convert_temp,
        text,
    )
    text = re.sub(r"(\d+\s*°C)\s*\(\s*\d+\s*°?\s*C\s*\)", r"\1", text)
    return text


def _parse_llm_recipe(text: str) -> Dict:
    def field(name):
        match = re.search(rf"{name}:\s*(.+)", text)
        return match.group(1).strip().strip("*").strip() if match else ""

    before_steps, _, after_steps = text.partition("Steps:")
    ingredient_block = before_steps.partition("Ingredients:")[2]

    ingredient_lines = []
    for line in ingredient_block.splitlines():
        line = re.sub(r"^[-*]\s*", "", line.strip())
        if line and not re.match(r"^(title|serves|ingredients)\s*:", line, re.I):
            ingredient_lines.append(line)

    steps = []
    for line in after_steps.splitlines():
        line = re.sub(rf"^{_STEP_NUMBER}\s*", "", line.strip())
        line = re.sub(r"^[-*]\s*", "", line)
        if line and not re.match(r"^(title|serves|ingredients)\s*:", line, re.I):
            steps.append(line)

    return {
        "title": field("Title"),
        "serves": field("Serves"),
        "ingredient_lines": ingredient_lines,
        "steps": steps,
    }


def generate_recipe_llm(
    ingredients: List[str], retriever, llm, context_docs: int = 3
) -> Dict:
    query = ", ".join(ingredients)
    candidates = retriever.invoke(query)
    docs = rerank_by_coverage(candidates, ingredients, top_n=context_docs)
    context = build_context(docs, max_docs=context_docs)

    prompt = LLM_PROMPT.format(ingredients=query, context=context)
    raw = llm([
        {"role": "system", "content": LLM_SYSTEM},
        {"role": "user", "content": prompt},
    ])[0]["generated_text"]

    parsed = _parse_llm_recipe(to_metric(raw))

    return {
        "title": parsed["title"],
        "serves": parsed["serves"],
        "ingredients": list(ingredients),
        "ingredient_lines": parsed["ingredient_lines"] or list(ingredients),
        "steps": parsed["steps"],
        "raw": raw,
        "sources": docs,
        "prompt": prompt,
    }

def format_recipe(result: Dict) -> str:
    header = result["title"] or ", ".join(
        i.title() for i in result["ingredients"]
    )
    serves = f"Serves: {result['serves']}\n" if result["serves"] else ""
    lines = result["ingredient_lines"]
    return (
        f"{header}\n{serves}\n"
        "Ingredients:\n" + "\n".join(f"- {line}" for line in lines) + "\n\n"
        "Instructions:\n"
        + "\n".join(f"{i}. {s}" for i, s in enumerate(result["steps"], 1))
    )
