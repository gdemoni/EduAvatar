import argparse
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


def load_vectorstore(index_dir: Path, model_name: str):
    print(f"DEBUG: Trying to load vectorstore from: {index_dir.absolute()}")
    if not index_dir.exists():
        print(f"DEBUG: Directory does NOT exist!")
        raise FileNotFoundError(f"index_dir not found: {index_dir}")
    
    # WORKAROUND: FAISS C++ implementation might fail with non-ASCII paths on Windows.
    # We change the current working directory to the parent of index_dir and use a relative path.
    original_cwd = os.getcwd()
    try:
        os.chdir(index_dir.parent)
        relative_index_dir = index_dir.name
        
        print(f"DEBUG: Changed CWD to {os.getcwd()}, loading relative path: {relative_index_dir}")
        
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return FAISS.load_local(
            relative_index_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    finally:
        os.chdir(original_cwd)


def main():
    # Get the project root directory (parent of the 'rag' directory)
    project_root = Path(__file__).resolve().parent.parent
    default_index_dir = project_root / "faiss_store"

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", default=str(default_index_dir))
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-zh-v1.5",
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("-k", type=int, default=4)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    vectorstore = load_vectorstore(Path(args.index_dir), args.model)
    # Perform the search
    # docs = vectorstore.similarity_search(args.query, k=args.k)
    docs_and_scores = vectorstore.similarity_search_with_score(args.query, k=args.k)

    print(f"\nResults for query: '{args.query}'\n")
    print("-" * 50)
    
    # for i, d in enumerate(docs, start=1):
    for i, (d, score) in enumerate(docs_and_scores, start=1):
        source = d.metadata.get("source", "Unknown")
        print(f"Result {i}:")
        print(f"Score: {score:.4f} (Lower is better for L2 distance)")
        print(f"Source: {source}")
        print(f"Content: {d.page_content.strip()}")
        print("-" * 50)


if __name__ == "__main__":
    main()
