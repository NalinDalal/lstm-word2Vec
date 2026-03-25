"""
Training script for Day 4 experiments:
- Train at 10k steps, compare outputs
- Train at 100k steps if time allows
- Document gradient norm behavior
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from char_lstm import CharLSTM, preprocess_text, create_sequences, generate


def train_steps(model, data, max_steps, lr, print_every=100, log_gradients=True):
    """
    Train for a specific number of optimization steps.

    Returns:
        losses: List of loss values at each logging step
        grad_norms: List of gradient norms at each logging step
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    losses = []
    grad_norms = []

    np.random.shuffle(data)

    total_steps = 0
    epoch = 0
    data_idx = 0

    while total_steps < max_steps:
        if data_idx >= len(data):
            np.random.shuffle(data)
            data_idx = 0
            epoch += 1

        batch = data[data_idx : data_idx + 32]
        if len(batch) < 2:
            data_idx = 0
            continue

        batch_size = len(batch)
        inputs = torch.LongTensor([seq[0] for seq in batch]).to(device)
        targets = torch.LongTensor([seq[1] for seq in batch]).to(device)

        optimizer.zero_grad()

        # Initialize hidden if needed
        if (
            not hasattr(model, "_hidden")
            or model._hidden is None
            or model._hidden[0].size(1) != batch_size
        ):
            h0 = torch.zeros(model.num_layers, batch_size, model.hidden_dim).to(device)
            c0 = torch.zeros(model.num_layers, batch_size, model.hidden_dim).to(device)
            hidden = (h0, c0)
        else:
            hidden = model._hidden

        outputs, hidden = model(inputs, hidden)
        model._hidden = (hidden[0].detach(), hidden[1].detach())

        loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
        loss.backward()

        if log_gradients and total_steps % print_every == 0:
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


def run_experiment():
    print("=" * 60)
    print("Day 4 Experiment: LSTM Training at Different Steps")
    print("=" * 60)

    # Load data
    with open("shakespeare.txt", "r") as f:
        text = f.read()

    print(f"\nDataset size: {len(text)} characters")

    chars, char_to_idx, idx_to_char = preprocess_text(text)
    print(f"Vocab size: {len(chars)}")

    seq_length = 50
    data = create_sequences(text, char_to_idx, seq_length)
    print(f"Number of sequences: {len(data)}")

    # Calculate approximate steps per epoch
    steps_per_epoch = len(data) // 32
    print(f"Approx steps per epoch: {steps_per_epoch}")

    # ===== BASELINE: Untrained model =====
    print("\n" + "=" * 60)
    print("BASELINE: Untrained model output")
    print("=" * 60)

    model = CharLSTM(
        vocab_size=len(chars), embedding_dim=256, hidden_dim=512, num_layers=2
    )

    output = generate(model, "ROMEO:", char_to_idx, idx_to_char, length=200)
    print("Untrained output:")
    print(output)

    # ===== TRAIN TO 10K STEPS =====
    print("\n" + "=" * 60)
    print("TRAINING TO 10,000 STEPS")
    print("=" * 60)

    model_10k = CharLSTM(
        vocab_size=len(chars), embedding_dim=256, hidden_dim=512, num_layers=2
    )

    losses_10k, grad_norms_10k = train_steps(
        model_10k, data, max_steps=10000, lr=0.001, print_every=500
    )

    print("\n--- Output at 10k steps ---")
    output_10k = generate(model_10k, "ROMEO:", char_to_idx, idx_to_char, length=200)
    print(output_10k)

    # Save gradient norms for analysis
    print("\n--- Gradient Norm Summary (10k) ---")
    print(f"Initial grad norm: {grad_norms_10k[0]:.4f}" if grad_norms_10k else "N/A")
    print(f"Final grad norm: {grad_norms_10k[-1]:.4f}" if grad_norms_10k else "N/A")
    print(
        f"Average grad norm: {np.mean(grad_norms_10k):.4f}" if grad_norms_10k else "N/A"
    )

    # ===== TRAIN TO 100K STEPS (if time allows) =====
    print("\n" + "=" * 60)
    print("TRAINING TO 100,000 STEPS")
    print("=" * 60)

    model_100k = CharLSTM(
        vocab_size=len(chars), embedding_dim=256, hidden_dim=512, num_layers=2
    )

    losses_100k, grad_norms_100k = train_steps(
        model_100k, data, max_steps=100000, lr=0.001, print_every=5000
    )

    print("\n--- Output at 100k steps ---")
    output_100k = generate(model_100k, "ROMEO:", char_to_idx, idx_to_char, length=200)
    print(output_100k)

    # ===== GRADIENT NORM ANALYSIS =====
    print("\n" + "=" * 60)
    print("GRADIENT NORM BEHAVIOR ANALYSIS")
    print("=" * 60)

    print("\n--- 100k Training Gradient Norms ---")
    print(f"Total gradient recordings: {len(grad_norms_100k)}")
    if grad_norms_100k:
        print(f"Min grad norm: {min(grad_norms_100k):.4f}")
        print(f"Max grad norm: {max(grad_norms_100k):.4f}")
        print(f"Mean grad norm: {np.mean(grad_norms_100k):.4f}")
        print(f"Std grad norm: {np.std(grad_norms_100k):.4f}")

        # Check if gradients are vanishing (shouldn't happen with LSTM)
        vanishing_count = sum(1 for g in grad_norms_100k if g < 0.1)
        print(
            f"Steps with grad norm < 0.1 (vanishing): {vanishing_count}/{len(grad_norms_100k)}"
        )

        # Check if gradients are exploding
        exploding_count = sum(1 for g in grad_norms_100k if g > 5.0)
        print(
            f"Steps with grad norm > 5.0 (exploding): {exploding_count}/{len(grad_norms_100k)}"
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key Observations:
1. LSTM gates allow gradients to flow unchanged through time
2. Gradient norms should remain relatively stable (not vanish)
3. The model learns to generate coherent text over training
4. Compare outputs: random -> word-like -> sentence-like -> Shakespeare-like
""")


if __name__ == "__main__":
    run_experiment()
