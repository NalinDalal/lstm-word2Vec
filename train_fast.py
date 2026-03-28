import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import random


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


def build_vocab(texts, min_freq=2):
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


# Use a larger corpus - Wikipedia or text8
# Let's use a simple approach: download text8 or use PTB with more data
# Actually, let's just use a larger window and more epochs on PTB

from data import load_data, get_vocab_map

print("Loading PTB data...")
train_data = load_data("./data/ptb.train.txt")
vocab_map = get_vocab_map()
idx_to_word = {idx: word for word, idx in vocab_map.items()}
words = [idx_to_word[i.item()] for i in train_data]

# Use smaller vocab by increasing min_freq - captures more common words better
texts = [words]
vocab, idx_to_word = build_vocab(texts, min_freq=5)
print(f"Vocabulary size: {len(vocab)}")

# Make sure our target words exist
for w in ["king", "man", "woman", "queen"]:
    if w not in vocab:
        print(f"Warning: {w} not in vocab!")

pairs = generate_training_data(texts, vocab, window_size=5)
print(f"Training pairs: {len(pairs)}")

EMBEDDING_DIM = 200
EPOCHS = 5
BATCH_SIZE = 8192

vocab_size = len(vocab)
word_freq = Counter([p[0] for p in pairs])
weights = np.array(
    [word_freq.get(i, 0) ** 0.75 for i in range(vocab_size)], dtype=np.float32
)
weights = weights / (weights.sum() + 1e-10)

model = Word2Vec(vocab_size, EMBEDDING_DIM)
optimizer = optim.Adam(model.parameters(), lr=0.025)

print(f"\nTraining Word2Vec with {EPOCHS} epochs, dim={EMBEDDING_DIM}...")
for epoch in range(EPOCHS):
    random.shuffle(pairs)
    total_loss = 0
    num_batches = 0

    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i : i + BATCH_SIZE]
        batch_size_actual = len(batch)
        if batch_size_actual < 2:
            continue

        targets = torch.tensor([p[0] for p in batch], dtype=torch.long)
        contexts = torch.tensor([p[1] for p in batch], dtype=torch.long)

        neg_indices = np.random.choice(
            vocab_size, size=(batch_size_actual, 10), p=weights
        )
        negatives = torch.tensor(neg_indices, dtype=torch.long)

        optimizer.zero_grad()
        loss = model(targets, contexts, negatives)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / num_batches:.4f}")

print("\nVerifying analogy...")
king = vocab.get("king")
man = vocab.get("man")
woman = vocab.get("woman")
queen = vocab.get("queen")

if not all([king, man, woman, queen]):
    print(f"Missing words! king={king}, man={man}, woman={woman}, queen={queen}")
else:
    embeddings = model.target_embeddings.weight.detach().numpy()
    result = embeddings[king] - embeddings[man] + embeddings[woman]
    distances = np.linalg.norm(embeddings - result, axis=1)
    exclude = {king, man, woman}
    valid_indices = [i for i in range(len(distances)) if i not in exclude]
    filtered_distances = distances[valid_indices]
    sorted_indices = np.argsort(filtered_distances)
    closest_idx = [valid_indices[i] for i in sorted_indices[:5]]

    print("\n=== king - man + woman ≈ ? ===")
    print(f"Expected: queen")
    closest_words = [idx_to_word[idx] for idx in closest_idx]
    print(f"Top 5: {closest_words}")

    if closest_words[0] == "queen":
        print("\n✓ SUCCESS!")
