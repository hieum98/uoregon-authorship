"""Load HRS evaluation data (TA1 and TA2) from hiatus-metrics format."""

import json
from ast import literal_eval
from pathlib import Path
from typing import List, Tuple, Union

import datasets
import numpy as np
import pandas as pd


def _normalize_author_ids(author_ids) -> tuple:
    if isinstance(author_ids, (list, tuple)):
        return tuple(sorted(str(a) for a in author_ids))
    return (str(author_ids),)


def load_ta1(
    query_path: str,
    candidate_path: str,
    ground_truth_path: str,
    text_key: str = "fullText",
    document_id_key: str = "documentID",
) -> Tuple[datasets.Dataset, datasets.Dataset, List[List[int]]]:
    """Load TA1 (document-level) data.

    Returns:
        (queries, candidates, ground_truth_positions)
        where ground_truth_positions[i] = candidate indices with same author as query i.
    """
    doc_to_author = {}
    with open(ground_truth_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get(document_id_key) or row.get("documentID"))
            author_ids = row.get("authorIDs", row.get("author_id", []))
            doc_to_author[doc_id] = _normalize_author_ids(author_ids)

    def _load_jsonl(path):
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    q_df = _load_jsonl(query_path)
    c_df = _load_jsonl(candidate_path)

    def _prepare(df):
        df = df.copy()
        if document_id_key in df.columns and document_id_key != "document_id":
            df = df.rename(columns={document_id_key: "document_id"})
        if text_key in df.columns and text_key != "text":
            df = df.rename(columns={text_key: "text"})
        df["author_id"] = df["document_id"].astype(str).map(
            lambda d: doc_to_author.get(d, (None,))
        )
        return df

    q_df = _prepare(q_df)
    c_df = _prepare(c_df)

    cols = ["text", "document_id", "author_id"]
    queries = datasets.Dataset.from_pandas(q_df[cols].copy(), preserve_index=False)
    candidates = datasets.Dataset.from_pandas(c_df[cols].copy(), preserve_index=False)

    c_authors = [c_df["author_id"].iloc[j] for j in range(len(c_df))]
    ground_truth_positions = []
    for i in range(len(q_df)):
        q_author = q_df["author_id"].iloc[i]
        ground_truth_positions.append([j for j, ca in enumerate(c_authors) if ca == q_author])

    return queries, candidates, ground_truth_positions


def load_ta2(
    query_path: str,
    candidate_path: str,
    ground_truth_path: str,
    ground_truth_query_labels_path: str,
    ground_truth_candidate_labels_path: str,
    text_key: str = "fullText",
    document_id_key: str = "documentID",
) -> Tuple[datasets.Dataset, datasets.Dataset, np.ndarray]:
    """Load TA2 (author-level) data.

    Returns:
        (queries, candidates, ground_truth)
        queries/candidates have text as List[str] per author.
        ground_truth is (n_queries, n_candidates) binary matrix.
    """
    def _load_labels(path):
        labels = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    labels.append(literal_eval(line.strip()))
        return labels

    query_labels = _load_labels(ground_truth_query_labels_path)
    candidate_labels = _load_labels(ground_truth_candidate_labels_path)
    ground_truth = np.load(ground_truth_path)

    def _load_jsonl(path):
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    q_df = _load_jsonl(query_path)
    c_df = _load_jsonl(candidate_path)

    if "authorIDs" in q_df.columns:
        q_df["_key"] = q_df["authorIDs"].map(_normalize_author_ids)
    else:
        raise ValueError("TA2 query data must have authorIDs column")
    if "authorSetIDs" in c_df.columns:
        c_df["_key"] = c_df["authorSetIDs"].map(_normalize_author_ids)
    else:
        raise ValueError("TA2 candidate data must have authorSetIDs column")

    text_col = text_key if text_key in q_df.columns else "fullText"
    doc_col = document_id_key if document_id_key in q_df.columns else "documentID"

    q_rows = []
    for ql in query_labels:
        key = _normalize_author_ids(ql)
        subset = q_df[q_df["_key"] == key]
        if subset.empty:
            raise ValueError(f"No query docs for label {ql}")
        q_rows.append({
            "text": subset[text_col].tolist(),
            "document_id": subset[doc_col].astype(str).tolist(),
            "author_id": key,
        })

    c_rows = []
    for cl in candidate_labels:
        key = _normalize_author_ids(cl)
        subset = c_df[c_df["_key"] == key]
        if subset.empty:
            raise ValueError(f"No candidate docs for label {cl}")
        c_rows.append({
            "text": subset[text_col].tolist(),
            "document_id": subset[doc_col].astype(str).tolist(),
            "author_id": key,
        })

    return (
        datasets.Dataset.from_list(q_rows),
        datasets.Dataset.from_list(c_rows),
        ground_truth.astype(np.float64),
    )
