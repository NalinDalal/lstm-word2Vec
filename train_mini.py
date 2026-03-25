"""Day 4: Mini demo"""

import torch
import torch.nn as nn
import torch.optim as optim
from char_lstm import CharLSTM, preprocess_text, create_sequences, generate

# Small subset
with open("shakespeare.txt", "r") as f:
    text = f.read()[:10000]
print(f"Text: {len(text)} chars")

chars, char_to_idx, idx_to_char = preprocess_text(text)
data = create_sequences(text, char_to_idx, 30)
print(f"Seqs: {len(data)}")

# Untrained
model = CharLSTM(vocab_size=len(chars), embedding_dim=32, hidden_dim=64, num_layers=1)
print("\n=== UNTRAINED ===")
print(generate(model, "ROMEO:", char_to_idx, idx_to_char, length=60))

# Train 200 steps
print("\n=== TRAINING 200 STEPS ===")
model = CharLSTM(vocab_size=len(chars), embedding_dim=32, hidden_dim=64, num_layers=1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

for step in range(200):
    batch = data[step % 100 * 20 : (step % 100 * 20) + 20]
    if len(batch) < 2:
        continue
    inputs = torch.LongTensor([seq[0] for seq in batch])
    targets = torch.LongTensor([seq[1] for seq in batch])

    optimizer.zero_grad()
    h0 = torch.zeros(1, len(batch), 64)
    c0 = torch.zeros(1, len(batch), 64)
    outputs, _ = model(inputs, (h0, c0))
    loss = criterion(outputs.view(-1, len(chars)), targets.view(-1))
    loss.backward()

    if step % 50 == 0:
        g = (
            sum(
                p.grad.norm().item() ** 2
                for p in model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )
        print(f"Step {step}: loss={loss.item():.4f}, grad={g:.4f}")

    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

print("\n=== AFTER 200 ===")
print(generate(model, "ROMEO:", char_to_idx, idx_to_char, length=60))
