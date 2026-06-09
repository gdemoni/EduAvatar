import argparse
import os
import glob
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import (UnstructuredWordDocumentLoader,)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".doc"}


def load_documents(source_dir: Path, glob_pattern: str):
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir not found: {source_dir}")

    matched_files = []
    for p in glob.glob(str(source_dir / glob_pattern), recursive=True):
        path = Path(p)
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            matched_files.append(path)

    documents = []
    load_errors = []

    for path in matched_files:
        suffix = path.suffix.lower()
        try:
            if suffix in {".txt", ".md"}:
                documents.extend(TextLoader(str(path), encoding="utf-8").load())
            elif suffix == ".pdf":
                documents.extend(PyPDFLoader(str(path)).load()) 

            elif suffix == ".docx":
                documents.extend(Docx2txtLoader(str(path)).load())

            elif suffix == ".doc":
                documents.extend(UnstructuredWordDocumentLoader(str(path)).load())
                
        except Exception as e:
            load_errors.append(f"{path}: {type(e).__name__}: {e}")

    if load_errors:
        raise ValueError("failed to load some files:\n" + "\n".join(load_errors))
    if not documents:
        raise ValueError(f"no documents matched: {source_dir} ({glob_pattern})")

    return documents


def build_vectorstore(
    source_dir: Path,
    index_dir: Path,
    glob_pattern: str,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,):
    print(f"[1/4] 加载文档: {source_dir}")
    documents = load_documents(source_dir, glob_pattern)
    print(f"      共加载 {len(documents)} 个文档")

    print(f"[2/4] 文本切分 (chunk_size={chunk_size}, overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"      共生成 {len(chunks)} 个文本片段")

    print(f"[3/4] 向量化 (模型: {model_name})")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print(f"[4/4] 保存索引到 {index_dir}")
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    print(f"      知识库构建完成！")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="D:/ai大模型应用开发/师智分身/后端/data")
    parser.add_argument("--index_dir", default="faiss_store")
    parser.add_argument("--glob", default="**/*")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-zh-v1.5",
    )
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    build_vectorstore(
        source_dir=Path(args.source_dir),
        index_dir=Path(args.index_dir),
        glob_pattern=args.glob,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
