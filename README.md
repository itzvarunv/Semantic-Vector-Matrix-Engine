#  From Phonetics to Geometry — Teaching a Machine to Understand Language

This repository contains two connected experiments in building a machine that understands human language from scratch — no pre-trained models, no massive corpora, no borrowed weights. Just mathematics, geometry, and the idea that meaning can be learned from patterns.

---

## The Problem Being Solved

Computers are good at finding things. But for decades they could only find things they had seen *exactly before*.

Search for "car repair" and you'd miss every article that used the word "automobile" or "vehicle" instead. Ask a database a question and it would scan for your exact words, not your intent. The machine had no idea that "heart attack" and "cardiac arrest" refer to the same thing, or that someone searching for "something to eat in Italy" probably wants results about Italian food.

This is the fundamental limitation of **keyword search** — it matches strings, not ideas.

```
Traditional search:   "car repair"  ──►  finds "car repair"
                                    ──✗  misses "auto mechanic"
                                    ──✗  misses "vehicle maintenance"
                                    ──✗  misses "fixing your automobile"
```

The problem being solved here is: **how do you teach a machine what words mean, so it can match intent rather than just letters?**

---

## Keyword Search vs. Semantic Search

The difference comes down to one distinction:

> **Traditional search finds matching words. Semantic search finds matching meaning.**

Consider a user who types: *"I feel terrible and my chest hurts"*

A keyword search engine looks for documents containing those exact words. It might return nothing useful, or worse, return unrelated results that happen to contain "chest" and "hurts."

A semantic search engine understands that this phrase lives in the same conceptual neighbourhood as "heart attack symptoms", "cardiac pain", "chest tightness causes" — not because the words match, but because the *meaning* is geometrically close in vector space.

| | Keyword Search | Semantic Search |
|---|---|---|
| How it works | Matches exact characters | Matches vector proximity |
| Finds synonyms | ✗ No | ✓ Yes |
| Understands context | ✗ No | ✓ Yes |
| "car" finds "automobile" | ✗ No | ✓ Yes |
| "fix" finds "repair" | ✗ No | ✓ Yes |
| Speed | Very fast | Depends on index size |
| Needs training data | ✗ No | ✓ Yes |

Semantic search is only possible if words have been turned into vectors first — if the machine has already learned, from reading, that "car" and "automobile" belong in the same region of meaning-space. That is exactly what this project builds.

---

## Why Semantic Search Matters

Every system that processes human language hits this wall eventually.

A customer support bot that only does keyword matching will fail the moment a user phrases their problem differently than the FAQ was written. A recommendation engine that matches on category tags will miss connections a human reader would find obvious. A medical database that requires exact terminology will fail patients who don't know the clinical words for what they're experiencing.

The deeper goal of this project is to build the *foundation* that makes semantic understanding possible: a vector space where proximity means relatedness, learned entirely from how words are actually used by humans — not from rules someone wrote down.

Once words are vectors, almost everything becomes geometry:
- **Search** becomes: find vectors near this query vector
- **Recommendation** becomes: find items near this user's taste vector  
- **Classification** becomes: which cluster does this document's centroid fall into
- **Translation** becomes: find the corresponding vector in another language's space

The hard part — and what this project tackles — is building that vector space honestly, from scratch, using nothing but mathematics and observed language patterns.

---

## The Core Idea

Human language is slippery. Words don't just carry definitions — they carry *relationships*. "Engine" and "car" belong together. "Knife" and "wife" do not, even though they rhyme. Teaching a machine to feel that difference without simply memorising a dictionary requires a different kind of representation.

The answer is **vectors** — turning every word into a point in multi-dimensional space. If two words end up close together in that space, the machine has learned they are related. If they are far apart, they are not. Distance *is* meaning.

This project explores two approaches to building that space, starting from a naive first attempt and evolving into a more principled system.

---

## Chapter 1 — Autocorrect.py (The Phonetic Era)

The first attempt used **phonetics** as a shortcut to geometry.

The intuition was reasonable: words that sound alike are often spelled alike, and misspellings tend to produce words that still *sound* like the intended word. So each character in a word was assigned a value based on its phonetic group — vowels got one score, labials another, sibilants another — turning each word into a numeric vector based purely on how it sounds.

```
a, e, i, o, u, y  →  5   (vowels / glides)
b, p, v, f        →  10  (labials)
c, s, z, x        →  15  (sibilants)
d, t, l, n, m, r  →  20  (liquids / dentals)
g, k, q, j        →  25  (gutturals)
h, w              →  2   (rare sounds)
```

A Soundex hash then filtered candidates to the same phonetic bucket before cosine similarity picked the best match. Type `recieve` and it finds `receive`. Type `nife` and it finds `knife`.

**It worked well for spelling correction.** But it had a fundamental flaw for the bigger goal of semantic understanding.

### The Problem with Phonetics

Because vectors were built from character sounds, words with *similar sounds* ended up geometrically close — regardless of meaning:

- `wife` and `knife` became neighbours in the vector space (they rhyme)
- `engine` and `car` lived in completely separate regions (they sound nothing alike)

This is fine for a spell-checker. It is a dead end for a system that needs to understand context.

---

## Chapter 2 — Word Vectoring Pro (The Semantic Era)

The second system threw out phonetics entirely and replaced them with **randomness and convergence**.

### The New Approach

Every new word that enters the system gets a completely random 15-dimensional vector — a random point in a 15-axis coordinate space. No phonetic shape. No positional encoding. Just noise.

```python
[round(random.uniform(-1.0, 1.0), 4) for _ in range(15)]
```

That random initialisation means `wife` and `knife` start in arbitrary, unrelated positions. So do `engine` and `car`. The slate is clean.

The meaning is then *learned from context*.

### Learning Through Co-occurrence

When a block of text is ingested, the system finds which words appear most densely — the words that recur frequently enough to be considered "dominant" in that passage. The threshold adapts to text length:

| Text length | Density threshold | Reasoning |
|---|---|---|
| ≤ 150 tokens | 2.5% | Short entries need strong repetition |
| 151–800 tokens | 1.5% | Standard blog/article length |
| > 800 tokens | 0.5% | Long essays; even moderate repetition counts |

Every pair of dominant words in a passage gets their vectors **nudged toward each other** — by a tiny fraction (2%) of the gap between them. One article barely moves them. But after hundreds of automotive articles, `engine` and `car` have been co-dominant so many times that their vectors have slowly drifted into the same neighbourhood.

`wife` and `knife` never co-dominate the same passages, so they never converge. The geometry reflects reality.

### Stop-Word Detection

A word that connects to *everything* — appearing as dominant in automotive articles, cooking blogs, finance pieces, and health content alike — is probably structural glue, not meaningful vocabulary ("the", "is", "said"). The system tracks unique co-token pairings per word via a ledger, and flags any word exceeding 100 unique connections as a stop-word candidate for the user to review.

### Clustering

Once training is complete, `grouper()` sweeps the entire vector space using cosine similarity. Words whose vectors point in roughly the same direction (≥ 70% similarity) get grouped into semantic clusters with a confidence score. The result is a map of meaning:

```json
{
  "r1": { "engine": 91.23, "car": 89.71, "horsepower": 87.44 },
  "r2": { "recipe": 88.10, "ingredient": 85.33, "bake": 82.90 }
}
```

No one told it these words were related. It learned it from reading.

---

## Repository Structure

```
├── Autocorrect.py              # Stage 1 — phonetic spell-checker
├── Word vectoring pro.py       # Stage 2 — semantic vector engine
├── testing.py                  # Automated RSS training harness
├── memory.json                 # Autocorrect's learned word database
├── semantic_brain.json         # Word Vectoring Pro's vector state
├── relations_history.json      # Latest semantic cluster snapshot
└── words.txt                   # Seed word list
```

---

## Why This Matters

Both systems are small. Neither will rival a transformer model. But they were built to answer a specific question: **can a machine learn that words are related by observing how humans use them, using nothing but basic mathematics?**

The answer, it turns out, is yes — as long as you give up on shortcuts like phonetics and let the geometry emerge from the data itself. Random initialisation + contextual convergence + cosine clustering is a miniature, interpretable version of the same core intuition behind word embeddings in modern NLP.

This project is the foundation of something larger: an algorithm that understands human language not because it was told the rules, but because it learned the patterns.

---

## Requirements

```bash
pip install feedparser jellyfish
```

**Autocorrect.py** — standalone, run directly:
```bash
python Autocorrect.py
```

**Word Vectoring Pro** — interactive mode:
```bash
python "Word vectoring pro.py"
```

**Automated training** across 5 RSS niches (Technology, Automotive, Culinary, Finance, Health):
```bash
python testing.py
```
