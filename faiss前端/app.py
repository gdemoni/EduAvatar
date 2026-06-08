import glob
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = Path(os.environ.get("FAISS_INDEX_DIR", str(BASE_DIR.parent / "faiss_store")))
EMBEDDING_MODEL = os.environ.get("FAISS_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
CHUNK_SIZE = int(os.environ.get("FAISS_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("FAISS_CHUNK_OVERLAP", "100"))
ALLOWED_UPLOAD_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".doc", ".pptx", ".ppt"}


def _load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _load_vectorstore(index_dir: Path, embeddings):
    if not index_dir.exists():
        return None
    if not (index_dir / "index.faiss").exists() or not (index_dir / "index.pkl").exists():
        return None
    original_cwd = os.getcwd()
    try:
        os.chdir(index_dir.parent)
        return FAISS.load_local(index_dir.name, embeddings, allow_dangerous_deserialization=True)
    finally:
        os.chdir(original_cwd)


def _split_text_to_docs(text: str, source: str, chunk_size: int | None = None, chunk_overlap: int | None = None):
    chunk_size = CHUNK_SIZE if chunk_size is None else int(chunk_size)
    chunk_overlap = CHUNK_OVERLAP if chunk_overlap is None else int(chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    raw_docs = [Document(page_content=text.strip(), metadata={"source": source.strip() or "manual_input"})]
    return splitter.split_documents(raw_docs)


def _split_docs(docs, chunk_size: int | None = None, chunk_overlap: int | None = None):
    chunk_size = CHUNK_SIZE if chunk_size is None else int(chunk_size)
    chunk_overlap = CHUNK_OVERLAP if chunk_overlap is None else int(chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def _load_ppt_loader(file_path: str):
    try:
        from langchain_community.document_loaders import UnstructuredPowerPointLoader
        return UnstructuredPowerPointLoader(file_path)
    except Exception:
        from langchain_community.document_loaders import UnstructuredFileLoader
        return UnstructuredFileLoader(file_path)


def _load_file_documents(file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return TextLoader(str(file_path), encoding="utf-8").load()
    if suffix == ".pdf":
        return PyPDFLoader(str(file_path)).load()
    if suffix == ".docx":
        return Docx2txtLoader(str(file_path)).load()
    if suffix == ".doc":
        return UnstructuredWordDocumentLoader(str(file_path)).load()
    if suffix in {".pptx", ".ppt"}:
        return _load_ppt_loader(str(file_path)).load()
    raise ValueError(f"不支持的文件类型: {suffix}")


def _load_documents_from_dir(source_dir: Path, glob_pattern: str):
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir not found: {source_dir}")

    matched_files = []
    for p in glob.glob(str(source_dir / glob_pattern), recursive=True):
        path = Path(p)
        if path.is_file() and path.suffix.lower() in ALLOWED_UPLOAD_SUFFIXES:
            matched_files.append(path)

    loaded_docs = []
    errors = []
    for path in matched_files:
        try:
            docs = _load_file_documents(path)
            for d in docs:
                metadata = dict(d.metadata or {})
                metadata["source"] = metadata.get("source") or path.name
                d.metadata = metadata
            loaded_docs.extend(docs)
        except Exception as e:
            errors.append(f"{path}: {type(e).__name__}: {e}")

    return loaded_docs, errors, len(matched_files)


def _upsert_chunks(chunks):
    embeddings = _load_embeddings()
    vectorstore = _load_vectorstore(INDEX_DIR, embeddings)
    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        vectorstore.add_documents(chunks)
    _save_vectorstore(vectorstore, INDEX_DIR)
    return _get_stats()


def _save_vectorstore(vectorstore: FAISS, index_dir: Path):
    index_dir.mkdir(parents=True, exist_ok=True)
    original_cwd = os.getcwd()
    try:
        os.chdir(index_dir.parent)
        vectorstore.save_local(index_dir.name)
    finally:
        os.chdir(original_cwd)


def _get_stats():
    embeddings = _load_embeddings()
    vectorstore = _load_vectorstore(INDEX_DIR, embeddings)
    if vectorstore is None:
        return {"exists": False, "total_vectors": 0}
    return {"exists": True, "total_vectors": int(vectorstore.index.ntotal)}


def _list_documents(vectorstore, source_filter: str | None = None):
    docs = list(vectorstore.docstore._dict.values())
    if source_filter:
        keyword = source_filter.strip().lower()
        if keyword:
            docs = [d for d in docs if keyword in str(d.metadata.get("source", "")).lower()]
    return docs


@app.get("/")
def index():
    return render_template("index.html", index_dir=str(INDEX_DIR), model=EMBEDDING_MODEL, stats=_get_stats())


@app.post("/api/ingest")
def ingest():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    source = str(payload.get("source", "")).strip() or "manual_input"

    if not text:
        return jsonify({"ok": False, "message": "文本不能为空"}), 400

    docs = _split_text_to_docs(text, source)
    if not docs:
        return jsonify({"ok": False, "message": "可写入的文本为空"}), 400

    stats = _upsert_chunks(docs)
    return jsonify(
        {
            "ok": True,
            "message": f"写入成功，新增分块 {len(docs)} 条",
            "added_chunks": len(docs),
            "total_vectors": stats["total_vectors"],
            "index_dir": str(INDEX_DIR),
        }
    )


@app.post("/api/upload")
def upload_files():
    files = request.files.getlist("files")
    source_prefix = str(request.form.get("source", "")).strip()
    if not files:
        return jsonify({"ok": False, "message": "请先拖拽或选择文件"}), 400

    loaded_docs = []
    errors = []
    processed_files = 0

    with tempfile.TemporaryDirectory(prefix="faiss_upload_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for i, f in enumerate(files, start=1):
            original_name = str(f.filename or "").strip()
            safe_name = Path(original_name).name
            if not safe_name:
                safe_name = f"upload_{i}"
            suffix = Path(safe_name).suffix.lower()
            if not suffix and original_name:
                suffix = Path(original_name).suffix.lower()
                safe_name = f"{safe_name}{suffix}"
            filename = safe_name
            if not filename:
                continue
            if suffix not in ALLOWED_UPLOAD_SUFFIXES:
                errors.append(f"{filename}: 不支持的文件类型")
                continue

            saved_path = tmp_path / filename
            f.save(saved_path)

            try:
                docs = _load_file_documents(saved_path)
                final_source = source_prefix or filename
                for d in docs:
                    metadata = dict(d.metadata or {})
                    metadata["source"] = final_source
                    d.metadata = metadata
                loaded_docs.extend(docs)
                processed_files += 1
            except Exception as e:
                errors.append(f"{filename}: {type(e).__name__}: {e}")

    if not loaded_docs:
        return jsonify({"ok": False, "message": "没有可写入的文件内容", "errors": errors}), 400

    chunks = _split_docs(loaded_docs)
    if not chunks:
        return jsonify({"ok": False, "message": "文件内容为空，无法分块", "errors": errors}), 400

    stats = _upsert_chunks(chunks)
    return jsonify(
        {
            "ok": True,
            "message": f"上传成功，处理文件 {processed_files} 个，新增分块 {len(chunks)} 条",
            "processed_files": processed_files,
            "added_chunks": len(chunks),
            "total_vectors": stats["total_vectors"],
            "index_dir": str(INDEX_DIR),
            "errors": errors,
        }
    )


@app.post("/api/build_from_dir")
def build_from_dir():
    payload = request.get_json(silent=True) or {}
    source_dir = str(payload.get("source_dir", "")).strip()
    if not source_dir:
        return jsonify({"ok": False, "message": "source_dir 不能为空"}), 400

    glob_pattern = str(payload.get("glob", "**/*")).strip() or "**/*"
    chunk_size = payload.get("chunk_size")
    chunk_overlap = payload.get("chunk_overlap")
    mode = str(payload.get("mode", "replace")).strip().lower()

    docs, errors, matched_files = _load_documents_from_dir(Path(source_dir), glob_pattern)
    if not docs:
        return jsonify({"ok": False, "message": "没有可写入的文件内容", "errors": errors}), 400

    chunks = _split_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return jsonify({"ok": False, "message": "文件内容为空，无法分块", "errors": errors}), 400

    if mode == "append":
        stats = _upsert_chunks(chunks)
    else:
        embeddings = _load_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        _save_vectorstore(vectorstore, INDEX_DIR)
        stats = _get_stats()

    return jsonify(
        {
            "ok": True,
            "message": f"处理文件 {matched_files} 个，新增分块 {len(chunks)} 条",
            "processed_files": matched_files,
            "added_chunks": len(chunks),
            "total_vectors": stats["total_vectors"],
            "index_dir": str(INDEX_DIR),
            "errors": errors,
            "mode": mode,
        }
    )


@app.get("/api/knowledge/list")
def list_knowledge():
    source = str(request.args.get("source", "")).strip()
    try:
        offset = int(request.args.get("offset", 0))
        limit = int(request.args.get("limit", 50))
    except ValueError:
        return jsonify({"ok": False, "message": "offset/limit 必须是数字"}), 400

    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    embeddings = _load_embeddings()
    vectorstore = _load_vectorstore(INDEX_DIR, embeddings)
    if vectorstore is None:
        return jsonify({"ok": True, "total": 0, "items": []})

    docs = _list_documents(vectorstore, source_filter=source)
    total = len(docs)
    items = []
    for d in docs[offset: offset + limit]:
        items.append(
            {
                "content": d.page_content,
                "source": str(d.metadata.get("source", "")),
                "metadata": dict(d.metadata or {}),
            }
        )
    return jsonify({"ok": True, "total": total, "items": items})


@app.get("/api/knowledge/search")
def search_knowledge():
    query = str(request.args.get("query", "")).strip()
    if not query:
        return jsonify({"ok": False, "message": "query 不能为空"}), 400

    try:
        k = int(request.args.get("k", 5))
    except ValueError:
        return jsonify({"ok": False, "message": "k 必须是数字"}), 400

    k = max(1, min(k, 20))

    embeddings = _load_embeddings()
    vectorstore = _load_vectorstore(INDEX_DIR, embeddings)
    if vectorstore is None:
        return jsonify({"ok": True, "results": []})

    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
    results = []
    for doc, score in docs_and_scores:
        results.append(
            {
                "content": doc.page_content,
                "source": str(doc.metadata.get("source", "")),
                "metadata": dict(doc.metadata or {}),
                "score": float(score),
            }
        )
    return jsonify({"ok": True, "results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8787, debug=False)
