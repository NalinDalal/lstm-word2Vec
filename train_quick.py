import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import random
from data import load_data, get_vocab_map
from word2vec import Word2Vec, build_vocab, generate_training_data

print("Loading PTB data...")
train_data = load_data("./data/ptb.train.txt")
vocab_map = get_vocab_map()
idx_to_word = {idx: word for word, idx in vocab_map.items()}
words = [idx_to_word[i.item()] for i in train_data]
texts = [words]
print(f"Loaded {len(words)} words")

vocab, idx_to_word = build_vocab(texts, min_freq=3)
print(f"Vocabulary size: {len(vocab)}")

EMBEDDING_DIM = 100
EPOCHS = 20
BATCH_SIZE = 4096
LR = 0.01

pairs = generate_training_data(texts, vocab, window_size=5)
print(f"Training pairs: {len(pairs)}")

vocab_size = len(vocab)
word_freq = Counter([p[0] for p in pairs])
weights = np.array(
    [word_freq.get(i, 0) ** 0.75 for i in range(vocab_size)], dtype=np.float32
)
weights = weights / (weights.sum() + 1e-10)

model = Word2Vec(vocab_size, EMBEDDING_DIM)
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"\nTraining Word2Vec with {EPOCHS} epochs...")
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
            vocab_size, size=(batch_size_actual, 5), p=weights
        )
        negatives = torch.tensor(neg_indices, dtype=torch.long)

        optimizer.zero_grad()
        loss = model(targets, contexts, negatives)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / num_batches:.4f}")

torch.save(model.state_dict(), "word2vec_model.pt")
print("Model saved!")

print("\nVerifying analogy...")
king = vocab.get("king")
man = vocab.get("man")
woman = vocab.get("woman")
queen = vocab.get("queen")

print(f"Word indices: king={king}, man={man}, woman={woman}, queen={queen}")

if not all([king, man, woman, queen]):
    print("Warning: Some words not in vocabulary!")
else:
    embeddings = model.target_embeddings.weight.detach().numpy()
    result = embeddings[king] - embeddings[man] + embeddings[woman]
    distances = np.linalg.norm(embeddings - result, axis=1)
    exclude = {king, man, woman}
    valid_indices = [i for i in range(len(distances)) if i not in exclude]
    filtered_distances = distances[valid_indices]
    sorted_indices = np.argsort(filtered_distances)
    closest_idx = [valid_indices[i] for i in sorted_indices[:5]]

    print("\n=== Verification: king - man + woman ≈ ? ===")
    print(f"Expected: queen")
    closest_words = [idx_to_word[idx] for idx in closest_idx]
    dist_list = [round(float(distances[idx]), 4) for idx in closest_idx]
    print(f"Top 5 closest words: {closest_words}")
    print(f"Distances: {dist_list}")

    if closest_words[0] == "queen":
        print("\n✓ SUCCESS: Analogy verified!")
    elif "queen" in closest_words:
        print(f"\n~ Partial: 'queen' is at position {closest_words.index('queen') + 1}")
    else:
        print(f"\n✗ 'queen' not in top 5")
