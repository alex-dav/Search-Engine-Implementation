from typing import List, Sequence

from nltk.corpus import wordnet as wn

def wordnet_candidates(term: str, max_synsets: int = 1, max_candidates: int = 6) -> List[str]:
    """Return candidate expansion terms from NLTK WordNet."""
    candidates: List[str] = []

    for synset in wn.synsets(term, pos=wn.NOUN)[:max_synsets]:
        for lemma in synset.lemmas():
            word = lemma.name().replace("_", " ").lower()
            if word != term.lower() and word not in candidates:
                candidates.append(word)
            if len(candidates) >= max_candidates:
                return candidates

    return candidates


def expand_query_wordnet(
    query_terms: Sequence[str],
    max_terms_per_word: int = 2,
) -> List[str]:
    """Optional extension: return a small WordNet-assisted expansion."""
    # Preserve original terms and cap additions to reduce uncontrolled query drift.
    expanded: List[str] = list(query_terms)

    for term in query_terms:
        candidates = wordnet_candidates(term)
        for candidate in candidates[:max_terms_per_word]:
            if candidate not in expanded:
                expanded.append(candidate)

    return expanded
