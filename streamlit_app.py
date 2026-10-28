import os
import time

import streamlit as st

import recipe_rag as rr

st.set_page_config(page_title="AI Recipe Generator", page_icon="🍳")
st.title("🍳 AI Recipe Generator")
st.caption(
    "RAG: FAISS retrieves real recipes as context, "
    f"{rr.LLM_MODEL} writes the recipe"
)

if not os.path.isdir(rr.INDEX_DIR):
    st.warning("FAISS index missing — run `python build_index.py` first.")


@st.cache_resource(show_spinner="Loading the FAISS index…")
def load_retrieval():
    store = rr.FAISS.load_local(
        rr.INDEX_DIR, rr.load_embeddings(), allow_dangerous_deserialization=True
    )
    return rr.get_retriever(store)


@st.cache_resource(show_spinner="Loading the generation model (about 1 min)…")
def load_model():
    return rr.load_llm()


retriever = load_retrieval()
llm = load_model()

st.markdown("### Your ingredients")
ingredients_input = st.text_input(
    "Comma-separated",
    placeholder="eggs, potatoes, cheese",
)

if st.button("Generate recipe", type="primary") and ingredients_input.strip():
    ingredients = [i.strip() for i in ingredients_input.split(",") if i.strip()]

    with st.spinner("The model is writing the recipe…"):
        start = time.time()
        result = rr.generate_recipe_llm(ingredients, retriever, llm)
        elapsed = time.time() - start

    st.markdown(f"## 🍽️ {result['title']}")
    caption = f"Original recipe, written by the model in {elapsed:.0f}s"
    if result["serves"]:
        caption = f"Serves {result['serves']} · " + caption
    st.caption(caption)

    col_ing, col_steps = st.columns([1, 2])
    with col_ing:
        st.markdown("**Ingredients**")
        for line in result["ingredient_lines"]:
            st.markdown(f"- {line}")
    with col_steps:
        st.markdown("**Instructions**")
        st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(result["steps"], 1)))

    with st.expander("Real recipes used as RAG context"):
        st.caption(
            "These dataset recipes are not shown as the answer — they go into "
            "the prompt as technique examples."
        )
        for doc in result["sources"]:
            lines = doc.metadata.get("ingredient_lines") or doc.metadata[
                "normalized_ingredients"
            ]
            st.markdown(f"- **{doc.metadata['title']}** — {', '.join(lines)}")

    with st.expander("Prompt sent to the model"):
        st.text(result["prompt"])
