"""Tokenization utilities for embedder and reranker data pipelines."""

from typing import List, Optional, Tuple

from transformers import PreTrainedTokenizer


def tokenize_example(
    text,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 512,
    **kwargs,
):
    """Tokenize text (str or list of str) with configurable kwargs."""
    return tokenizer(
        text=text,
        max_length=max_seq_length,
        truncation=True,
        **kwargs,
    )


# ---- Reranker prompt formatting ----

SYSTEM_PROMPT = (
    "Judge whether the Document is written by the same author as the Query. "
    'Note that the answer can only be "yes" or "no".'
)

DEFAULT_INSTRUCTION = (
    "Given a query text, determine if the provided document was written by the same author"
)


def format_authorship_prompt(
    query_text: str,
    doc_text: str,
    instruction: Optional[str] = None,
) -> str:
    """Format a (query, doc) pair into the user-turn content string."""
    if instruction is None:
        instruction = DEFAULT_INSTRUCTION
    return (
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query_text}\n"
        f"<Document>: {doc_text}"
    )


def get_prefix_suffix_tokens(
    tokenizer: PreTrainedTokenizer,
) -> Tuple[List[int], List[int]]:
    """Get prefix (system+user start) and suffix (assistant think block) token IDs."""
    prefix = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
    )
    suffix = (
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
    return (
        tokenizer.encode(prefix, add_special_tokens=False),
        tokenizer.encode(suffix, add_special_tokens=False),
    )


def tokenize_pair(
    query_text: str,
    doc_text: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 1024,
    instruction: Optional[str] = None,
) -> dict:
    """Tokenize a (query, doc) pair for the reranker using prefix/suffix pattern."""
    prefix_tokens, suffix_tokens = get_prefix_suffix_tokens(tokenizer)
    user_content = format_authorship_prompt(query_text, doc_text, instruction)

    content_max = max_length - len(prefix_tokens) - len(suffix_tokens)
    content_tokens = tokenizer.encode(
        user_content, add_special_tokens=False,
        truncation=True, max_length=content_max,
    )
    return {"input_ids": prefix_tokens + content_tokens + suffix_tokens}
