"""
Day 4 Experiment: Train at 10k steps, compare outputs
Smaller model for faster training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from char_lstm import CharLSTM, preprocess_text, create_sequences, generate


def train_steps(model, data, max_steps, lr, print_every=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    losses = []
    grad_norms = []

    np.random.shuffle(data)

    total_steps = 0
    data_idx = 0

    while total_steps < max_steps:
        if data_idx >= len(data):
            np.random.shuffle(data)
            data_idx = 0

        batch = data[data_idx : data_idx + 32]
        if len(batch) < 2:
            data_idx = 0
            continue

        batch_size = len(batch)
        inputs = torch.LongTensor([seq[0] for seq in batch]).to(device)
        targets = torch.LongTensor([seq[1] for seq in batch]).to(device)

        optimizer.zero_grad()

        h0 = torch.zeros(model.num_layers, batch_size, model.hidden_dim).to(device)
        c0 = torch.zeros(model.num_layers, batch_size, model.hidden_dim).to(device)
        hidden = (h0, c0)

        outputs, hidden = model(inputs, hidden)
        hidden = (hidden[0].detach(), hidden[1].detach())

        loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
        loss.backward()

        if total_steps % print_every == 0:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            grad_norms.append(total_norm)
            losses.append(loss.item())
            print(
                f"Step {total_steps}: Loss = {loss.item():.4f}, Grad norm = {total_norm:.4f}"
            )

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_steps += 1
        data_idx += 32

    return losses, grad_norms


# Load data
print("Loading Shakespeare corpus...")
with open("shakespeare.txt", "r") as f:
    text = f.read()

print(f"Dataset size: {len(text)} characters")

chars, char_to_idx, idx_to_char = preprocess_text(text)
print(f"Vocab size: {len(chars)}")

data = create_sequences(text, char_to_idx, 50)
print(f"Sequences: {len(data)}")

# Baseline: untrained
print("\n" + "=" * 50)
print("BASELINE: Untrained model")
print("=" * 50)
model = CharLSTM(vocab_size=len(chars), embedding_dim=128, hidden_dim=256, num_layers=2)
output = generate(model, "ROMEO:", char_to_idx, idx_to_char, length=200)
print("Untrained output:")
print(output)

# Train 10k steps
print("\n" + "=" * 50)
print("TRAINING TO 10,000 STEPS")
print("=" * 50)
model_10k = CharLSTM(
    vocab_size=len(chars), embedding_dim=128, hidden_dim=256, num_layers=2
)
losses, grad_norms = train_steps(
    model_10k, data, max_steps=10000, lr=0.002, print_every=500
)

print("\n--- Output at 10k steps ---")
output_10k = generate(model_10k, "ROMEO:", char_to_idx, idx_to_char, length=200)
print(output_10k)

# Gradient analysis
print("\n" + "=" * 50)
print("GRADIENT NORM ANALYSIS")
print("=" * 50)
print(f"Total recordings: {len(grad_norms)}")
print(f"Min grad norm: {min(grad_norms):.4f}")
print(f"Max grad norm: {max(grad_norms):.4f}")
print(f"Mean grad norm: {np.mean(grad_norms):.4f}")
vanishing = sum(1 for g in grad_norms if g < 0.1)
print(f"Steps with grad < 0.1: {vanishing}/{len(grad_norms)}")
