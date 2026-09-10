"""Lightweight GraphRAG: builds an in-memory knowledge graph from the same
PDFs used for vector search, and surfaces relevant entity relationships as
extra context alongside the vector-retrieved chunks.

The graph is persisted to disk as JSON (via networkx) and built
incrementally, mirroring how the Chroma vector store tracks processed files.
"""
import os
import json
import logging
from typing import Dict, List, Optional

import networkx as nx
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer

logger = logging.getLogger(__name__)

GRAPH_DB_PATH = os.getenv("GRAPH_DB_PATH", "graph_db/knowledge_graph.json")
GRAPH_PROCESSED_TRACKER = os.getenv("GRAPH_PROCESSED_TRACKER", "graph_processed_files.txt")
GRAPH_MAX_CHUNKS_PER_RUN = int(os.getenv("GRAPH_MAX_CHUNKS_PER_RUN", "40"))
GRAPH_MAX_HOPS = int(os.getenv("GRAPH_MAX_HOPS", "1"))
GRAPH_MAX_CONTEXT_TRIPLES = int(os.getenv("GRAPH_MAX_CONTEXT_TRIPLES", "20"))

ALLOWED_NODES = [
    "Disease", "Symptom", "Medicine", "Treatment", "Remedy", "BodyPart",
    "Cause", "RiskFactor", "Complication", "Procedure",
]
ALLOWED_RELATIONSHIPS = [
    "HAS_SYMPTOM", "TREATED_BY", "CAUSES", "AFFECTS", "PRESCRIBED_FOR",
    "RISK_FACTOR_FOR", "LEADS_TO", "RELATED_TO", "PREVENTS", "DIAGNOSED_BY",
]


def _split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n• ", "\n- ", "\n", ". ", " "],
    )
    return splitter.split_documents(docs)


def _load_processed(path: str) -> set:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def _save_processed(path: str, names: set) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for name in sorted(names):
            f.write(name + "\n")


def _save_graph(graph: nx.MultiDiGraph) -> None:
    os.makedirs(os.path.dirname(GRAPH_DB_PATH) or ".", exist_ok=True)
    data = nx.node_link_data(graph)
    with open(GRAPH_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _load_graph() -> nx.MultiDiGraph:
    if os.path.exists(GRAPH_DB_PATH):
        with open(GRAPH_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return nx.node_link_graph(data, directed=True, multigraph=True)
    return nx.MultiDiGraph()


def _node_key(node_id) -> str:
    return str(node_id).strip().lower()


def _merge_graph_documents(graph: nx.MultiDiGraph, graph_documents) -> None:
    for gd in graph_documents:
        source = gd.source.metadata.get("source") if gd.source else None
        for node in gd.nodes:
            key = _node_key(node.id)
            if not key:
                continue
            if not graph.has_node(key):
                graph.add_node(key, label=str(node.id).strip(), type=node.type)
        for rel in gd.relationships:
            src_key = _node_key(rel.source.id)
            tgt_key = _node_key(rel.target.id)
            if not src_key or not tgt_key:
                continue
            if not graph.has_node(src_key):
                graph.add_node(src_key, label=str(rel.source.id).strip(), type=rel.source.type)
            if not graph.has_node(tgt_key):
                graph.add_node(tgt_key, label=str(rel.target.id).strip(), type=rel.target.type)
            graph.add_edge(src_key, tgt_key, type=rel.type, source=source)


async def build_or_load_graph_async(
    llm: BaseLanguageModel,
    knowledge_dir: str = "knowledge",
) -> nx.MultiDiGraph:
    """Load the persisted graph and extend it with any PDFs not yet ingested.

    Bounded by GRAPH_MAX_CHUNKS_PER_RUN so a slow/rate-limited LLM can't stall
    startup indefinitely; remaining files are picked up on the next run.
    """
    graph = _load_graph()
    processed = _load_processed(GRAPH_PROCESSED_TRACKER)

    if not os.path.isdir(knowledge_dir):
        return graph

    pdf_files = sorted(f for f in os.listdir(knowledge_dir) if f.lower().endswith(".pdf"))
    new_files = [f for f in pdf_files if f not in processed]
    if not new_files:
        return graph

    new_chunks: List[Document] = []
    fully_loaded_files = set()
    for pdf in new_files:
        loader = PyPDFLoader(os.path.join(knowledge_dir, pdf))
        pages = loader.load()
        chunks = _split_docs(pages)
        if len(new_chunks) + len(chunks) > GRAPH_MAX_CHUNKS_PER_RUN and new_chunks:
            break
        new_chunks.extend(chunks)
        fully_loaded_files.add(pdf)

    if not new_chunks:
        logger.info("GraphRAG: no chunks fit this run's budget; will resume next startup")
        return graph

    if len(new_chunks) > GRAPH_MAX_CHUNKS_PER_RUN:
        new_chunks = new_chunks[:GRAPH_MAX_CHUNKS_PER_RUN]

    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=ALLOWED_NODES,
        allowed_relationships=ALLOWED_RELATIONSHIPS,
    )

    try:
        graph_documents = await transformer.aconvert_to_graph_documents(new_chunks)
    except Exception as e:
        logger.warning("GraphRAG: extraction failed this run, keeping existing graph: %s", e)
        return graph

    _merge_graph_documents(graph, graph_documents)
    processed |= fully_loaded_files
    _save_processed(GRAPH_PROCESSED_TRACKER, processed)
    _save_graph(graph)
    logger.info(
        "GraphRAG: graph now has %d nodes / %d edges (%d new chunks from %d file(s))",
        graph.number_of_nodes(), graph.number_of_edges(), len(new_chunks), len(fully_loaded_files),
    )
    return graph


def _match_seed_nodes(graph: nx.MultiDiGraph, query: str, limit: int = 8) -> List[str]:
    q = (query or "").lower()
    if not q:
        return []
    matches = []
    for node_id, data in graph.nodes(data=True):
        label = (data.get("label") or node_id).lower()
        if label and (label in q or node_id in q):
            matches.append(node_id)
        if len(matches) >= limit:
            break
    return matches


def graph_context(graph: nx.MultiDiGraph, query: str) -> str:
    """Return newline-separated relationship triples relevant to `query`."""
    if graph is None or graph.number_of_nodes() == 0:
        return ""

    seed_nodes = _match_seed_nodes(graph, query)
    if not seed_nodes:
        return ""

    visited = set(seed_nodes)
    frontier = set(seed_nodes)
    triples = []

    for _ in range(max(GRAPH_MAX_HOPS, 1)):
        if len(triples) >= GRAPH_MAX_CONTEXT_TRIPLES:
            break
        next_frontier = set()
        for node_id in frontier:
            for _, tgt, data in graph.out_edges(node_id, data=True):
                triples.append((node_id, data.get("type", "RELATED_TO"), tgt))
                if tgt not in visited:
                    next_frontier.add(tgt)
            for src, _, data in graph.in_edges(node_id, data=True):
                triples.append((src, data.get("type", "RELATED_TO"), node_id))
                if src not in visited:
                    next_frontier.add(src)
        visited |= next_frontier
        frontier = next_frontier

    if not triples:
        return ""

    def label_of(n):
        return graph.nodes[n].get("label", n) if graph.has_node(n) else n

    lines = []
    seen = set()
    for s, rel, t in triples[:GRAPH_MAX_CONTEXT_TRIPLES]:
        line = f"{label_of(s)} --{rel}--> {label_of(t)}"
        if line not in seen:
            seen.add(line)
            lines.append(line)

    return "\n".join(lines)


def graph_context_document(graph: nx.MultiDiGraph, query: str) -> Optional[Document]:
    """Wrap graph_context() as a Document so it can be stuffed alongside
    vector-retrieved chunks in the QA chain's context."""
    text = graph_context(graph, query)
    if not text:
        return None
    return Document(
        page_content=f"Related facts from the knowledge graph:\n{text}",
        metadata={"source": "knowledge_graph"},
    )
