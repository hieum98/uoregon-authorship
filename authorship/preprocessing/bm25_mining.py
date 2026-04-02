"""BM25 hard negative mining for authorship attribution training data.

Usage:
    python -m authorship.preprocessing.bm25_mining \
        --dataset_name Hieuman/reddit_bm25 \
        --output_dir ./data/reddit_bm25 \
        --top_k 512
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional

import bm25s
import datasets
import numpy as np
from tqdm import tqdm


STOPWORD_LANGS = {
    "en": "english", "zh": None, "ar": "arabic", "ru": "russian",
}


class BM25Retriever:
    """BM25 retriever for mining hard negatives per language split."""

    def __init__(self):
        self.indices: Dict[str, bm25s.BM25] = {}
        self.corpora: Dict[str, List[str]] = {}

    def build_from_text(self, texts: List[str], language: str = "en", use_stopwords: bool = True) -> None:
        stopwords = STOPWORD_LANGS.get(language) if use_stopwords else None
        tokenized = bm25s.tokenize(texts, stopwords=stopwords if stopwords else None)
        index = bm25s.BM25()
        index.index(tokenized)
        self.indices[language] = index
        self.corpora[language] = texts

    def retrieve(self, query: str, language: str = "en", top_k: int = 512) -> List[int]:
        index = self.indices[language]
        tokenized = bm25s.tokenize([query], stopwords=STOPWORD_LANGS.get(language))
        results, scores = index.retrieve(tokenized, k=min(top_k, len(self.corpora[language])))

        row_scores = scores[0]
        row_ids = results[0].tolist()
        if len(row_scores) > 0:
            max_score = row_scores[0]
            threshold = 0.8 * max_score
            filtered = [rid for rid, s in zip(row_ids, row_scores) if s < threshold]
            return filtered if filtered else row_ids
        return row_ids

    def batch_retrieve(
        self, queries: List[str], language: str = "en", top_k: int = 512,
    ) -> List[List[int]]:
        index = self.indices[language]
        stopwords = STOPWORD_LANGS.get(language)
        tokenized = bm25s.tokenize(queries, stopwords=stopwords if stopwords else None)
        k = min(top_k, len(self.corpora[language]))
        results, scores = index.retrieve(tokenized, k=k)

        all_ids = []
        for i in range(len(queries)):
            row_scores = scores[i]
            row_ids = results[i].tolist()
            if len(row_scores) > 0:
                max_score = row_scores[0]
                threshold = 0.8 * max_score
                filtered = [rid for rid, s in zip(row_ids, row_scores) if s < threshold]
                all_ids.append(filtered if filtered else row_ids)
            else:
                all_ids.append(row_ids)
        return all_ids

    def save_index(self, path: str, language: str = "en"):
        self.indices[language].save(os.path.join(path, f"bm25_{language}"))

    def load_index(self, path: str, language: str = "en"):
        self.indices[language] = bm25s.BM25.load(os.path.join(path, f"bm25_{language}"))


def add_column_with_polars(
    dataset: datasets.Dataset,
    column_name: str,
    values: List,
) -> datasets.Dataset:

    import polars as pl

    # Convert once to pandas, then to polars
    pdf = dataset.to_pandas()
    df = pl.from_pandas(pdf)
    df = df.with_columns(pl.Series(column_name, values))
    # Convert back to pandas then to HF Dataset
    pdf_out = df.to_pandas()
    return datasets.Dataset.from_pandas(pdf_out, preserve_index=False)


def build_same_author_ids(dataset: datasets.Dataset) -> datasets.Dataset:
    """Add sameAuthor_docIDs column grouping documents by authorIDs."""
    import polars as pl

    df = pl.DataFrame({"idx": range(len(dataset)), "authorIDs": dataset["authorIDs"]})
    grouped = (
        df.group_by("authorIDs")
        .agg(pl.col("idx").alias("doc_indices"))
    )
    author_to_docs = dict(zip(
        grouped["authorIDs"].to_list(),
        grouped["doc_indices"].to_list(),
    ))

    doc_indices_col = [
        [d for d in author_to_docs[aid] if d != i]
        for i, aid in enumerate(dataset["authorIDs"])
    ]
    return add_column_with_polars(dataset, "sameAuthor_docIDs", doc_indices_col)


def mine_hard_negatives(
    dataset: datasets.Dataset,
    language: str = "en",
    top_k: int = 512,
    batch_size: int = 1000,
    use_stopwords: bool = True,
) -> datasets.Dataset:
    """Add BM25-mined hard_negative_docIDs and optionally sameAuthor_docIDs."""
    texts = dataset["fullText"]
    retriever = BM25Retriever()
    print(f"Building BM25 index for {language} ({len(texts)} docs)...")
    retriever.build_from_text(texts, language, use_stopwords)

    dataset = dataset.map(lambda _, idx: {"docID": idx}, with_indices=True, num_proc=1)

    if "sameAuthor_docIDs" not in dataset.column_names:
        print("Building sameAuthor_docIDs...")
        dataset = build_same_author_ids(dataset)

    all_negatives = []
    for start in tqdm(range(0, len(texts), batch_size), desc="BM25 retrieval"):
        batch = texts[start : start + batch_size]
        neg_ids = retriever.batch_retrieve(batch, language, top_k)
        all_negatives.extend(neg_ids)
    dataset = add_column_with_polars(dataset, "hard_negative_docIDs", all_negatives)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="BM25 hard negative mining")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--languages", nargs="+", default=["en"])
    parser.add_argument("--top_k", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--use_stopwords", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.dataset_name):
        hf_data = datasets.load_from_disk(args.dataset_name)
    else:
        hf_data = datasets.load_dataset(args.dataset_name)
    result = {}
    for lang in args.languages:
        if lang not in hf_data:
            print(f"Skipping {lang}: not in dataset")
            continue
        ds = hf_data[lang]
        ds = ds.remove_columns(["sameAuthor_docIDs"])
        ds = mine_hard_negatives(ds, language=lang, top_k=args.top_k, batch_size=args.batch_size, use_stopwords=args.use_stopwords)
        result[lang] = ds
        print(f"[{lang}] Mined negatives for {len(ds)} documents")
        breakpoint()

    combined = datasets.DatasetDict(result)
    combined.save_to_disk(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
