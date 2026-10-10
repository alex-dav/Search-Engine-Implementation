"""Preprocessing utility for Task 3 system experiments."""

from typing import List

from bs4 import BeautifulSoup
import unicodedata
import html
import re
from nltk.corpus import stopwords as nltk_stopwords

from nltk.tokenize import word_tokenize

_ENTITY_PATTERN = re.compile(r"&[a-zA-Z]+;")

__all__ = ["preprocess"]


def preprocess(raw_html_list: List[str]) -> List[List[str]]:
    """Clean and tokenize a batch of noisy documents.

    Steps:
        1. Strip HTML tags via BeautifulSoup.
        2. Unescape HTML entities and normalise unicode.
        3. Remove remaining tag fragments with a regex.
        4. Tokenize with ``nltk.word_tokenize``.
        5. Lower-case, drop stop-words, punctuation-only tokens, and
           tokens with length <= 2.
        6. Split hyphenated terms into parts while keeping the original.

    Args:
        raw_html_list: One raw HTML or noisy web-text string per document.

    Returns:
        One token list per input document, preserving input order and length.
        Return an empty list for a document with no usable text.
    """
    if not raw_html_list:
        return []

    stopwords = set(nltk_stopwords.words("english"))
    tokenised_docs = []
    for raw_html in raw_html_list:
        if not raw_html:
            tokenised_docs.append([])
            continue

        # Strip HTML tags and extract plain text
        soup = BeautifulSoup(raw_html, "html.parser")
        cleaned_text = soup.get_text(separator=" ")

        # Normalize text
        normalised_text = html.unescape(cleaned_text)
        normalised_text = _ENTITY_PATTERN.sub(" ", normalised_text)
        normalised_text = unicodedata.normalize("NFKC", normalised_text)
        # Remove any remaining HTML tag fragments
        normalised_text = re.sub(r"</?[a-zA-Z]+[^>]*>", " ", normalised_text)
        normalised_text = re.sub(r"\s+", " ", normalised_text).strip()

        # Tokenize
        word_tokens = word_tokenize(normalised_text)

        # Lowercase, remove stopwords, punctuation-only, short tokens
        filtered = []
        for token in word_tokens:
            t = token.lower()
            if t in stopwords:
                continue
            if not any(c.isalnum() for c in t):
                continue
            if len(t) <= 2:
                continue

            # Split hyphenated terms keep original AND add parts
            if "-" in t:
                parts = t.split("-")
                filtered.append(t)
                for part in parts:
                    if len(part) > 2 and any(c.isalnum() for c in part):
                        filtered.append(part)
            else:
                filtered.append(t)

        tokenised_docs.append(filtered)

    return tokenised_docs
