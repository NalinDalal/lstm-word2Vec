import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from data import load_data, get_vocab_map


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(
            self.target_embeddings.weight, -0.5 / embedding_dim, 0.5 / embedding_dim
        )
        nn.init.zeros_(self.context_embeddings.weight)

    def forward(self, target, context, negatives):
        target_emb = self.target_embeddings(target)
        context_emb = self.context_embeddings(context)
        pos_score = (target_emb * context_emb).sum(dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-10)

        neg_emb = self.context_embeddings(negatives)
        neg_score = torch.bmm(neg_emb, target_emb.unsqueeze(2)).squeeze()
        neg_loss = torch.log(torch.sigmoid(-neg_score) + 1e-10).sum(dim=1)

        return -(pos_loss + neg_loss).mean()


def build_vocab(texts, min_freq=5):
    word_counts = Counter()
    for words in texts:
        word_counts.update(words)

    vocab = {
        word: idx
        for idx, (word, count) in enumerate(
            sorted(
                [(w, c) for w, c in word_counts.items() if c >= min_freq],
                key=lambda x: -x[1],
            )
        )
    }
    idx_to_word = {idx: word for word, idx in vocab.items()}
    return vocab, idx_to_word


def generate_training_data(texts, vocab, window_size=5):
    pairs = []
    for words in texts:
        indices = [vocab[w] for w in words if w in vocab]
        for i, center in enumerate(indices):
            context_start = max(0, i - window_size)
            context_end = min(len(indices), i + window_size + 1)
            for j in range(context_start, context_end):
                if i != j:
                    pairs.append((center, indices[j]))
    return pairs


def get_negative_samples(pairs, vocab, num_samples=5, exponent=0.75):
    word_counts = Counter()
    for center, _ in pairs:
        word_counts[center] += 1

    vocab_size = len(vocab)
    weights = np.zeros(vocab_size)
    for idx, count in word_counts.items():
        weights[idx] = count**exponent
    weights = weights / weights.sum()

    negatives = []
    for _ in range(num_samples):
        neg = np.random.choice(vocab_size, size=len(pairs), p=weights)
        negatives.append(neg)
    return np.array(negatives).T


def train_word2vec(texts, vocab, embedding_dim=100, epochs=5, batch_size=2048, lr=0.01):
    pairs = generate_training_data(texts, vocab)
    print(f"Training pairs: {len(pairs)}")

    vocab_size = len(vocab)
    word_freq = Counter([p[0] for p in pairs])
    weights = np.array(
        [word_freq.get(i, 0) ** 0.75 for i in range(vocab_size)], dtype=np.float32
    )
    weights = weights / (weights.sum() + 1e-10)

    model = Word2Vec(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss = 0
        num_batches = 0

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            batch_size_actual = len(batch)
            if batch_size_actual < 2:
                continue

            targets = torch.tensor([p[0] for p in batch], dtype=torch.long)
            contexts = torch.tensor([p[1] for p in batch], dtype=torch.long)

            neg_indices = np.random.choice(
                vocab_size, size=(batch_size_actual, 5), p=weights
            )
            negatives = torch.tensor(neg_indices, dtype=torch.long)

            optimizer.zero_grad()
            loss = model(targets, contexts, negatives)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_batches:.4f}")

    torch.save(model.state_dict(), "word2vec_model.pt")
    print("Model saved to word2vec_model.pt")

    return model


def plot_embeddings(model, idx_to_word, vocab, num_words=100):
    embeddings = model.target_embeddings.weight.detach().numpy()

    words = list(vocab.keys())[:num_words]
    indices = [vocab[w] for w in words]
    word_embeddings = embeddings[indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, num_words - 1))
    embeddings_2d = tsne.fit_transform(word_embeddings)

    plt.figure(figsize=(12, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
    for i, word in enumerate(words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)
    plt.title("Word Embeddings (t-SNE)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig("word_embeddings_tsne.png", dpi=150)
    plt.close()
    print("t-SNE plot saved to word_embeddings_tsne.png")


def verify_analogy(model, vocab, idx_to_word):
    king = vocab.get("king")
    man = vocab.get("man")
    woman = vocab.get("woman")
    queen = vocab.get("queen")

    if not all([king, man, woman, queen]):
        print("Warning: Some words not in vocabulary")
        print(f"king: {king}, man: {man}, woman: {woman}, queen: {queen}")
        return

    embeddings = model.target_embeddings.weight.detach().numpy()

    result = embeddings[king] - embeddings[man] + embeddings[woman]

    distances = np.linalg.norm(embeddings - result, axis=1)

    exclude = {king, man, woman}
    valid_indices = [i for i in range(len(distances)) if i not in exclude]
    filtered_distances = distances[valid_indices]
    sorted_indices = np.argsort(filtered_distances)
    closest_idx = [valid_indices[i] for i in sorted_indices[:5]]

    print("\nVerification: king - man + woman ≈ ?")
    print(f"Expected: queen")
    closest_words = [idx_to_word[idx] for idx in closest_idx]
    dist_list = [round(float(distances[idx]), 4) for idx in closest_idx]
    print(f"Closest words: {closest_words}")
    print(f"Distances: {dist_list}")


def main():
    print("Loading PTB data...")
    train_data = load_data("./data/ptb.train.txt")
    vocab_map = get_vocab_map()
    idx_to_word = {idx: word for word, idx in vocab_map.items()}
    words = [idx_to_word[i.item()] for i in train_data]
    texts = [words]
    print(f"Loaded {len(words)} words")

    vocab, idx_to_word = build_vocab(texts, min_freq=3)
    print(f"Vocabulary size: {len(vocab)}")

    print("\nTraining Word2Vec (Skip-gram with Negative Sampling)...")
    model = train_word2vec(
        texts, vocab, embedding_dim=64, epochs=10, batch_size=4096, lr=0.005
    )

    print("\nPlotting embeddings with t-SNE...")
    plot_embeddings(model, idx_to_word, vocab, num_words=150)

    print("\nVerifying king - man + woman ≈ queen...")
    verify_analogy(model, vocab, idx_to_word)


if __name__ == "__main__":
    main()
