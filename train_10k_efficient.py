"""Day 4: Train at 10k steps"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from char_lstm import CharLSTM, preprocess_text, create_sequences, generate

with open("shakespeare.txt", "r") as f:
    text = f.read()[:30000]
print(f"Text: {len(text)} chars")

chars, char_to_idx, idx_to_char = preprocess_text(text)
data = create_sequences(text, char_to_idx, 35)
print(f"Seqs: {len(data)}")

# Untrained
model = CharLSTM(vocab_size=len(chars), embedding_dim=48, hidden_dim=96, num_layers=1)
print("\n=== UNTRAINED ===")
print(generate(model, "ROMEO:", char_to_idx, idx_to_char, length=80))

# Train 10k steps
print("\n=== TRAINING 10K STEPS ===")
model = CharLSTM(vocab_size=len(chars), embedding_dim=48, hidden_dim=96, num_layers=1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

grad_norms = []
np.random.shuffle(data)
batch_size = 32

for step in range(10000):
    batch_idx = (step * batch_size) % (len(data) - batch_size)
    batch = data[batch_idx : batch_idx + batch_size]
    if len(batch) < 2:
        continue

    inputs = torch.LongTensor([seq[0] for seq in batch])
    targets = torch.LongTensor([seq[1] for seq in batch])

    optimizer.zero_grad()
    h0 = torch.zeros(1, batch_size, 96)
    c0 = torch.zeros(1, batch_size, 96)
    outputs, _ = model(inputs, (h0, c0))
    loss = criterion(outputs.view(-1, len(chars)), targets.view(-1))
    loss.backward()

    if step % 1000 == 0:
        g = (
            sum(
                p.grad.norm().item() ** 2
                for p in model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )
        grad_norms.append(g)
        print(f"Step {step}: loss={loss.item():.4f}, grad={g:.4f}")

    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

print("\n=== AFTER 10K ===")
print(generate(model, "ROMEO:", char_to_idx, idx_to_char, length=200))

print("\n=== GRADIENT ANALYSIS ===")
print(f"Total recordings: {len(grad_norms)}")
print(
    f"Min: {min(grad_norms):.4f}, Max: {max(grad_norms):.4f}, Mean: {np.mean(grad_norms):.4f}, Std: {np.std(grad_norms):.4f}"
)
vanishing = sum(1 for g in grad_norms if g < 0.1)
exploding = sum(1 for g in grad_norms if g > 5.0)
print(f"Grad < 0.1 (vanishing): {vanishing}/{len(grad_norms)}")
print(f"Grad > 5.0 (exploding): {exploding}/{len(grad_norms)}")
