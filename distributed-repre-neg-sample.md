# [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546)

This paper extends the Skip-gram model from the original Word2Vec paper with several key improvements that enable faster training and better quality word vectors.

## Main Contributions

1. **Subsampling of frequent words** - significant speedup + better representations for rare words
2. **Negative Sampling** - simplified alternative to hierarchical softmax
3. **Phrase embeddings** - learning representations for multi-word phrases
4. **Demonstration of additive compositionality** - vector arithmetic works for phrases too

---

## Skip-gram Model Recap

Given a sequence of training words $w_1, w_2, …, w_T$, the Skip-gram model maximizes:

$$\frac{1}{T}\sum_{t=1}^{T}\sum_{-c \le j \le c, j \neq 0} \log p(w_{t+j} | w_t)$$

Where:
- $c$ = size of training context (window size)
- Larger $c$ = more training examples = higher accuracy (at cost of training time)

The basic Skip-gram uses softmax:

$$p(w_O | w_I) = \frac{\exp(v_{w_O}'^\top v_{w_I})}{\sum_{w=1}^W \exp(v_w'^\top v_{w_I})}$$

Problem: Computing the denominator (partition function) is $O(V)$ where $V$ is vocabulary size - too slow for large vocabularies.

---

## Extension 1: Subsampling of Frequent Words

In very large corpora, the most frequent words (e.g., "the", "in", "a") can occur hundreds of millions of times.

These words provide less information than rare words, and training on them is wasteful.

**Solution**: Randomly discard words with probability:

$$P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}$$

Where:
- $t$ = chosen threshold (typically $10^{-5}$)
- $f(w_i)$ = frequency of word $w_i$

**Effect**:
- ~2x to 10x speedup
- Significantly improves accuracy for rare words
- More regular word representations overall

Intuition: Common words appear very frequently but contribute less to learning. Sub-sampling reduces their influence proportionally.

---

## Extension 2: Negative Sampling

Instead of using hierarchical softmax or full softmax, use a simplified version of **Noise Contrastive Estimation (NCE)**.

### Why NCE?

For each training example (center word, context word pair), instead of computing probability over entire vocabulary, we only need to distinguish the true context word from $k$ random "noise" samples.

### Negative Sampling Objective

For each $(w_I, w_O)$ pair, maximize:

$$\log \sigma(v_{w_O}'^\top v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} [\log \sigma(-v_{w_i}'^\top v_{w_I})]$$

Where:
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ (sigmoid)
- $k$ = number of negative samples (typically 5-20)
- $P_n(w)$ = noise distribution

### The Key: Unigram Distribution Raised to 3/4 Power

The noise distribution is critical. They use:

$$P_n(w) = \frac{U(w)^{3/4}}{Z}$$

Where:
- $U(w)$ = unigram frequency distribution
- $Z$ = normalization constant

**Why 3/4?**

Empirically works better than other options. The 3/4 power reduces the probability of very frequent words being sampled as negatives (too easy to distinguish), while still giving them higher probability than rare words.

Example frequencies: [0.9, 0.03, 0.07]
After 3/4: [0.92, 0.06, 0.05]  (more balanced)

---

## Comparison: Hierarchical Softmax vs Negative Sampling

| Aspect | Hierarchical Softmax | Negative Sampling |
|--------|---------------------|-------------------|
| Speed | Faster for frequent words | Faster overall |
| Quality | Good for rare words | Better for frequent words |
| Complexity | Binary tree traversal | Simple sampling |
| Memory | Needs tree structure | Just random sampling |

Negative sampling learns **better representations for frequent words** compared to hierarchical softmax.

---

## Extension 3: Phrase Embeddings

Inherent limitation: Word representations cannot represent idiomatic phrases.

Example: "Canada" + "Air" ≠ "Air Canada"

### Finding Phrases

Simple data-driven approach using unigram and bigram counts:

$$score(w_i, w_j) = \frac{\text{count}(w_i w_j) - \delta}{\text{count}(w_i) \times \text{count}(w_j)}$$

If score > threshold, combine into a phrase token (e.g., "New_York").

### Training Phrase Embeddings

Treat identified phrases as single tokens during training.

This allows learning:

$$\vec{\text{NewYork}} \approx \vec{\text{New}} + \vec{\text{York}}$$

or

$$\vec{\text{Paris}} - \vec{\text{France}} + \vec{\text{Germany}} \approx \vec{\text{Berlin}}$$

---

## Additive Compositionality

A remarkable property discovered in this paper:

> Simple vector arithmetic can often produce meaningful results

### Examples

semantic:
- $\vec{\text{Paris}} - \vec{\text{France}} + \vec{\text{Germany}} \approx \vec{\text{Berlin}}$
- $\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$

syntactic:
- $\vec{\text{bigger}} - \vec{\text{big}} + \vec{\text{small}} \approx \vec{\text{smaller}}$

### Why This Works

The Skip-gram training objective with softmax tries to **maximize the product of context probabilities**. This naturally leads to:

$$\vec{w}_I \cdot \vec{w}_O \approx \log P(w_O | w_I)$$

For words appearing in similar contexts:

$$P(w'|w) \approx P(w|w')$$
$$\vec{w} \cdot \vec{w}' \approx \vec{w}' \cdot \vec{w}$$

This symmetry leads to additive relationships being preserved in the vector space.

---

## Key Insights

1. **Subsampling + Negative Sampling** are complementary techniques that significantly improve training efficiency

2. **Unigram with 3/4 power** is empirically optimal for negative sampling distribution

3. **Phrases can be learned** by treating them as single tokens with appropriate detection

4. **Additive compositionality** emerges naturally from the Skip-gram objective - not explicitly trained, but a byproduct

5. **Linear relationships** in embedding space enable analogical reasoning without any supervised labels

---

## mental model

**Negative Sampling** replaces expensive softmax with simple binary classification:

```
Input: "king"
Output: "queen" (positive sample)
        + k random words (negative samples)

Training: Learn to distinguish true context from noise
```

Instead of learning $P(\text{context} | \text{word})$ over entire vocabulary, just learn to separate true pair from noise.

Why it works:
- Noise words are randomly sampled, so the model learns "this word doesn't go with that context"
- Frequent words appear more often as positive samples, so they get more training signal
- The 3/4 power balances this - gives rare words a fair chance

**Subsampling** removes redundant training examples:

```
Original: "the" appears 1,000,000 times
After subsampling: "the" appears ~10,000 times

Result: Faster training + more diverse word representations
```

---

## Implementation Notes

In our implementation:

```python
# Subsampling probability
P_discard = 1 - np.sqrt(threshold / freq)

# Negative sampling distribution  
P_noise = unigram_freq ** 0.75
P_noise = P_noise / P_noise.sum()
```

This differs from hierarchical softmax which uses binary tree traversal.

---

## Results Summary

On analogy tasks (semantic + syntactic):

- Skip-gram + Negative Sampling: ~70% accuracy
- Skip-gram + Hierarchical Softmax: ~60% accuracy  
- With subsampling: +10-15% improvement

Phrases can be learned with similar accuracy to individual words when they have sufficient occurrence in the corpus.

---

## Relation to Previous Paper (1301.3781)

This paper builds on "Efficient Estimation of Word Representations in Vector Space" by:

1. Adding practical improvements (subsampling, negative sampling)
2. Extending to phrases
3. Formalizing why additive compositionality works
4. Showing that simple models + huge data >> complex models + small data

[Implementation](./word2vec.py)
