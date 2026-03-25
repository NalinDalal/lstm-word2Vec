"""Day 4: Train at 10k steps - minimal version"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from char_lstm import CharLSTM, preprocess_text, create_sequences, generate

# Load data
with open("shakespeare.txt", "r") as f:
    text = f.read()
print(f"Text: {len(text)} chars")

chars, char_to_idx, idx_to_char = preprocess_text(text)
print(f"Vocab: {len(chars)}")

data = create_sequences(text, char_to_idx, 50)
print(f"Seqs: {len(data)}")

# Baseline
model = CharLSTM(vocab_size=len(chars), embedding_dim=64, hidden_dim=128, num_layers=1)
print("\n=== UNTRAINED ===")
print(generate(model, "ROMEO:", char_to_idx, idx_to_char, length=100))

# Train
print("\n=== TRAINING 10K STEPS ===")
model = CharLSTM(vocab_size=len(chars), embedding_dim=64, hidden_dim=128, num_layers=1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)
device = torch.device("cpu")
model = model.to(device)

grad_norms = []
np.random.shuffle(data)
total_steps = 0

while total_steps < 10000:
    for i in range(0, len(data), 32):
        if total_steps >= 10000:
            break
        batch = data[i : i + 32]
        if len(batch) < 2:
            continue
        inputs = torch.LongTensor([seq[0] for seq in batch]).to(device)
        targets = torch.LongTensor([seq[1] for seq in batch]).to(device)

        optimizer.zero_grad()
        h0 = torch.zeros(1, len(batch), 128).to(device)
        c0 = torch.zeros(1, len(batch), 128).to(device)
        outputs, _ = model(inputs, (h0, c0))
        loss = criterion(outputs.view(-1, len(chars)), targets.view(-1))
        loss.backward()

        if total_steps % 500 == 0:
            g = (
                sum(
                    p.grad.norm().item() ** 2
                    for p in model.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            grad_norms.append(g)
            print(f"Step {total_steps}: loss={loss.item():.4f}, grad={g:.4f}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_steps += 1

print("\n=== AFTER 10K ===")
print(generate(model, "ROMEO:", char_to_idx, idx_to_char, length=100))

print(f"\n=== GRADIENT ANALYSIS ===")
print(
    f"Min: {min(grad_norms):.4f}, Max: {max(grad_norms):.4f}, Mean: {np.mean(grad_norms):.4f}"
)
vanishing = sum(1 for g in grad_norms if g < 0.1)
print(f"Grad < 0.1: {vanishing}/{len(grad_norms)} (vanishing)")
