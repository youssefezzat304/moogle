from __future__ import annotations
from collections import Counter
from tokenizers import Tokenizer

_UNIGRAM_PREFIX = "\u2581"

def _normalize_token(token: str) -> str:
  return token.lstrip(_UNIGRAM_PREFIX).lower()

def _is_whole_token(tokens: list[str], term: str) -> bool:
  if len(tokens) != 1:
    return False
  
  return _normalize_token(tokens[0]) == term.lower()

def _std(values: list[float | int]) -> float:
  if len(values) < 2:
    return 0.0
  
  mean = sum(values) / len(values)
  variance = sum((x - mean) ** 2 for x in values) / len(values)
  return variance ** 0.5

def vocab_overlap(t1: Tokenizer, t2: Tokenizer) -> dict:
  v1 = set(_normalize_token(t) for t in t1.get_vocab().keys())
  v2 = set(_normalize_token(t) for t in t2.get_vocab().keys())
  
  shared = v1 & v2
  only_in_t1 = v1 - v2
  only_in_t2 = v2 - v1
  union = v1 | v2
  jaccard = len(shared) / len(union) if union else 0.0
  
  return {
    "vocab_size_t1"    : len(v1),
    "vocab_size_t2"    : len(v2),
    "shared"           : len(shared),
    "only_in_t1"       : len(only_in_t1),
    "only_in_t2"       : len(only_in_t2),
    "jaccard"          : round(jaccard, 4),
    "shared_tokens"    : sorted(shared),
    "only_in_t1_tokens": sorted(only_in_t1),
    "only_in_t2_tokens": sorted(only_in_t2),
  }

def fertility(tokenizer: Tokenizer, texts: list[str]) -> dict:
  if not texts:
      return {
        "fertility"           : 0.0,
        "mean_tokens_per_sent": 0.0,
        "std_tokens_per_sent" : 0.0,
        "min_tokens_per_sent" : 0,
        "max_tokens_per_sent" : 0,
        "total_tokens"        : 0,
        "total_words"         : 0,
        "num_sentences"       : 0,
      }
  
  token_counts: list[int | float] = []
  word_counts: list[int] = []
  
  for text in texts:
    encoded = tokenizer.encode(text)
    token_counts.append(len(encoded.ids))
    word_counts.append(len(text.split()))
    
  total_tokens = sum(token_counts)
  total_words = sum(word_counts)
  
  return {
        "fertility"           : round(total_tokens / total_words, 4) if total_words else 0.0,
        "mean_tokens_per_sent": round(sum(token_counts) / len(token_counts), 2),
        "std_tokens_per_sent" : round(_std(token_counts), 2),
        "min_tokens_per_sent" : min(token_counts),
        "max_tokens_per_sent" : max(token_counts),
        "total_tokens"        : total_tokens,
        "total_words"         : total_words,
        "num_sentences"       : len(texts),
    }
  
def unk_rate(tokenizer: Tokenizer, texts: list[str]) -> dict:
    unk_id = tokenizer.token_to_id("[UNK]")
    total = 0
    unks = 0

    for text in texts:
        ids = tokenizer.encode(text).ids
        total += len(ids)
        if unk_id is not None:
            unks += ids.count(unk_id)

    return {
        "unk_rate"    : round(unks / total, 6) if total else 0.0,
        "unk_count"   : unks,
        "total_tokens": total,
    }

def fragmentation_check(tokenizer: Tokenizer, terms: list[str]) -> list[dict]:
    results = []
    for term in terms:
        encoded = tokenizer.encode(term)
        tokens = encoded.tokens

        is_whole = _is_whole_token(tokens, term)

        results.append({
            "term"      : term,
            "tokens"    : tokens,
            "num_pieces": len(tokens),
            "is_whole"  : is_whole,
        })

    return results

def token_frequency(
    tokenizer: Tokenizer,
    texts: list[str],
    top_n: int = 30,
    skip_special: bool = True,
    normalize: bool = True,
) -> list[dict]:
    special : set[str] = {"[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"}
    counter : Counter  = Counter()

    for text in texts:
        encoded = tokenizer.encode(text)
        for token in encoded.tokens:
            if skip_special and token in special:
                continue
                
            if normalize:
                token = _normalize_token(token)
                
            counter[token] += 1

    return [
        {"rank": rank + 1, "token": token, "count": count}
        for rank, (token, count) in enumerate(counter.most_common(top_n))
    ]
