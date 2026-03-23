# LSTM + Word2Vec

**Why this week matters:** LSTM fixes vanishing gradients with gates. Word2Vec shows that words have geometry — "king − man + woman ≈ queen". These two ideas together make everything in weeks 11–15 possible.

**Papers — read in this order**

1. **[LANDMARK]** [Hochreiter & Schmidhuber (1997) — LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf) — gates solve the vanishing gradient problem
- [LSTM Implementation](https://github.com/wojzaremba/lstm)
- [Colah — Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — **mandatory**, read before or alongside the paper
- [ ] Project:  **Char-level LSTM Shakespeare generator** — train on Shakespeare corpus, compare outputs at 1k / 10k / 100k steps. Log gradient norms and confirm they no longer vanish.



2. [Mikolov et al. (2013) — Word2Vec: Efficient Estimation of Word Representations](https://arxiv.org/pdf/1301.3781)
- [ ] Project  **Word2Vec from scratch with analogy visualiser** — implement skip-gram with negative sampling. Plot embeddings with t-SNE. Verify "king − man + woman ≈ queen".


3. [Mikolov et al. (2013) — Distributed Representations + Negative Sampling](https://arxiv.org/pdf/1310.4546) · [implementation](https://github.com/SkalskiP/vlms-zero-to-hero/blob/master/01_natural_language_processing_fundamentals/01_01_word2vec_with_sub_sampling_and_negative_sampling_in_pytorch.ipynb)

**Supplementary papers** (read when curious, not blocking)
- [Neural Turing Machines](https://arxiv.org/abs/1410.5401) — memory-augmented networks, natural extension of LSTM thinking
- [Pointer Networks](https://papers.nips.cc/paper/5866-pointer-networks) — attention that copies from input rather than generating
- [RNN Regularisation (dropout for RNNs)](https://arxiv.org/abs/1409.2329)

**Resources**
- [NLP Playlist (Stanford CS224N)](https://youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z) — use throughout Phase 3

Lecture 1 — Introduction and Word Vectors
Lecture 2 — Word Vectors and Word Senses
Lecture 6 — Language Models and RNNs
Lecture 7 — Vanishing Gradients, Fancy RNNs ← this is literally LSTM


# 7-Day Plan:

Day 2 (Monday) — 3-4 hrs
- Finish LSTM paper with focused sections on gates
- Start: Char-level LSTM implementation setup
- Understand the data preprocessing

Day 3 (Tuesday) — 4-5 hrs
- Complete Char-level LSTM training code
- Run first training at 1k steps, log gradient norms
- Verify gradients aren't vanishing

Day 4 (Wednesday) — 4-5 hrs
- Train at 10k steps, compare outputs
- Train at 100k steps if time allows
- Document gradient norm behavior

Day 5 (Thursday) — 4-5 hrs
- Read Word2Vec papers (both Mikolov papers)
- Start skip-gram implementation

Day 6 (Friday) — 4-5 hrs
- Implement negative sampling
- Train Word2Vec embeddings
- Create t-SNE visualization

Day 7 (Saturday) — 4-5 hrs
- Verify "king − man + woman ≈ queen" analogy
- Finalize projects and documentation
- Catch up on any supplementary reading


