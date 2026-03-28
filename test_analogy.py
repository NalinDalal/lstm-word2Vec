import numpy as np
import torch
from collections import Counter
from data import load_data, get_vocab_map
from word2vec import Word2Vec, build_vocab

print("Loading PTB data...")
train_data = load_data("./data/ptb.train.txt")
vocab_map = get_vocab_map()
idx_to_word = {idx: word for word, idx in vocab_map.items()}
words = [idx_to_word[i.item()] for i in train_data]
texts = [words]
print(f"Loaded {len(words)} words")

vocab, idx_to_word = build_vocab(texts, min_freq=3)
print(f"Vocabulary size: {len(vocab)}")

print("\nLoading model...")
model = Word2Vec(len(vocab), 64)
model.load_state_dict(torch.load("word2vec_model.pt", map_location="cpu"))
model.eval()

king = vocab.get("king")
man = vocab.get("man")
woman = vocab.get("woman")
queen = vocab.get("queen")

print(f"\nWord indices: king={king}, man={man}, woman={woman}, queen={queen}")

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
    else:
        print(
            f"\n✗ Analogy not perfect. 'queen' is at position {closest_words.index('queen') + 1 if 'queen' in closest_words else 'not in top 5'}"
        )
