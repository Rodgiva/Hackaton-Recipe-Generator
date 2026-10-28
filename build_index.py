"""Build (or rebuild) the FAISS index over recipe ingredients.

    python build_index.py            # uses recipe_rag.SAMPLE_SIZE
    python build_index.py 50000      # or an explicit number of recipes

Rebuild whenever SAMPLE_SIZE, EMBEDDING_MODEL or build_documents change --
otherwise the cached index silently stays on the old data.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recipe_rag as rr


def human_size(path):
    if os.path.isdir(path):
        total = sum(
            os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)
        )
    else:
        total = os.path.getsize(path)
    return f"{total / 1_048_576:.1f} MB"


def main(n):
    print(f"Building the index for {n:,} recipes".replace(",", " "))
    print(f"  dataset    : {rr.DATASET_NAME}")
    print(f"  embeddings : {rr.EMBEDDING_MODEL}")
    print()

    t0 = time.time()
    rows = rr.load_recipe_sample(n=n)
    print(f"[1/3] streaming + filtering: {len(rows):,} recipes "
          f"in {(time.time()-t0)/60:.1f} min".replace(",", " "))

    t0 = time.time()
    documents = rr.build_documents(rows)
    print(f"[2/3] documents: {len(documents):,} ({time.time()-t0:.0f}s)"
          .replace(",", " "))

    t0 = time.time()
    rr.build_or_load_vectorstore(documents, force_rebuild=True)
    print(f"[3/3] FAISS index: {(time.time()-t0)/60:.1f} min "
          f"-> {human_size(rr.INDEX_DIR)}")

    print()
    t0 = time.time()
    store = rr.build_or_load_vectorstore([], force_rebuild=False)
    retriever = rr.get_retriever(store)
    print(f"Reload from cache: {time.time()-t0:.0f}s")

    for ings in (["eggs", "potatoes", "onions", "cheese"],
                 ["zucchini", "tomato", "garlic"],
                 ["salmon", "lemon", "butter"]):
        t0 = time.time()
        docs = rr.rerank_by_coverage(
            retriever.invoke(", ".join(ings)), ings, top_n=1
        )
        covered = rr.count_covered(docs[0], ings)
        print(f"  {str(ings):44s} {(time.time()-t0)*1000:4.0f} ms  "
              f"{covered}/{len(ings)}  {docs[0].metadata['title'][:45]}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else rr.SAMPLE_SIZE)
