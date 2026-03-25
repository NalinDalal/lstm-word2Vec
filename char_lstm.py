"""
Character-level LSTM for text generation.

This module implements a character-level language model using LSTM recurrent
neural networks. The model learns to predict the next character given a
sequence of previous characters, trained on the Shakespeare corpus.

Key components:
- CharLSTM: PyTorch LSTM model for character prediction
- preprocess_text: Build character vocabulary from text
- create_sequences: Create training data (input, target) pairs
- train: Training loop with gradient norm logging
- generate: Generate new text given a starting string

Why this demonstrates LSTM advantages:
- Gradient norms are logged to verify gradients don't vanish
- LSTM gates allow learning long-range dependencies (1000+ time steps)
- Compare outputs at 1k, 10k, 100k training steps
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


class CharLSTM(nn.Module):
    """
    Character-level LSTM language model.

    Architecture:
        Input (char index) -> Embedding -> LSTM (x2) -> Linear -> Output (probs)

    The LSTM's gated architecture allows gradients to flow unchanged through
    time when gates are open, solving the vanishing gradient problem that
    affects standard RNNs.

    Args:
        vocab_size: Number of unique characters in the vocabulary
        embedding_dim: Dimension of character embeddings (default: 256)
        hidden_dim: Dimension of LSTM hidden state (default: 512)
        num_layers: Number of LSTM layers (default: 2)

    Forward pass:
        Takes a batch of character indices (batch_size x seq_len)
        Returns logits for next character prediction and hidden state
    """

    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2):
        super(CharLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Convert character indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM processes sequence, maintaining hidden state across time steps
        # The gates (input, forget, output) control information flow
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)

        # Convert LSTM output to character probabilities
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass through the LSTM.

        Args:
            x: Tensor of character indices, shape (batch_size, seq_len)
            hidden: Tuple (h, c) of hidden and cell states, or None

        Returns:
            output: Logits for next character, shape (batch_size, seq_len, vocab_size)
            hidden: Updated (h, c) tuple for next time step
        """
        batch_size = x.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size)

        # Embedding converts integer indices to dense vectors
        embedded = self.embedding(x)

        # LSTM processes sequence, gates decide what info to keep/drop
        lstm_out, hidden = self.lstm(embedded, hidden)

        # Linear layer maps hidden state to character logits
        output = self.fc(lstm_out)
        return output, hidden

    def init_hidden(self, batch_size):
        """
        Initialize hidden state and cell state for LSTM.

        Both h (hidden state) and c (cell state) start as zeros.
        The cell state carries long-term information through the sequence.

        Args:
            batch_size: Number of sequences in batch

        Returns:
            Tuple (h0, c0) of zero tensors
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return (h0, c0)


def preprocess_text(text):
    """
    Build character vocabulary from text corpus.

    Creates bidirectional mappings between characters and indices.
    Sorting ensures reproducible vocabulary ordering.

    Args:
        text: String containing the training text

    Returns:
        chars: Sorted list of unique characters
        char_to_idx: Dict mapping character -> index
        idx_to_char: Dict mapping index -> character
    """
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return chars, char_to_idx, idx_to_char


def create_sequences(text, char_to_idx, seq_length):
    """
    Create training sequences for next-character prediction.

    For each position i, the input is characters [i, i+seq_length)
    and the target is characters [i+1, i+seq_length+1) shifted by 1.

    Example (seq_length=4):
        Input:  "Hell" -> Target: "ello"

    Args:
        text: Source text string
        char_to_idx: Character to index mapping
        seq_length: Number of characters in each input sequence

    Returns:
        List of (input_indices, target_indices) tuples
    """
    data = []
    for i in range(len(text) - seq_length):
        input_seq = text[i : i + seq_length]
        target_seq = text[i + 1 : i + seq_length + 1]
        input_indices = [char_to_idx[c] for c in input_seq]
        target_indices = [char_to_idx[c] for c in target_seq]
        data.append((input_indices, target_indices))
    return data


def download_shakespeare():
    """
    Download Shakespeare corpus for training.

    Uses the tiny Shakespeare dataset from Andrej Karpathy's char-rnn.
    Contains ~1MB of Shakespeare plays.

    Returns:
        Path to the downloaded/local file
    """
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = "shakespeare.txt"

    if not os.path.exists(filepath):
        import urllib.request

        print("Downloading Shakespeare corpus...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete!")
    return filepath


def train(model, data, epochs, lr, print_every=1000, log_gradients=True):
    """
    Train the character-level LSTM model.

    Uses cross-entropy loss and Adam optimizer. Gradient norms are logged
    to verify that LSTM gates prevent vanishing gradients - unlike standard
    RNNs where gradients decay exponentially through time.

    Args:
        model: CharLSTM model instance
        data: List of (input, target) sequence tuples
        epochs: Number of training epochs
        lr: Learning rate for Adam optimizer
        print_every: How often to log gradient norms
        log_gradients: Whether to compute and log gradient norms

    Returns:
        losses: List of average loss per epoch
        grad_norms: List of gradient norms at logging steps
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    losses = []
    grad_norms = []

    total_steps = 0
    for epoch in range(epochs):
        total_loss = 0
        hidden = None

        np.random.shuffle(data)

        # Process in batches of 32 sequences
        for i in range(0, len(data), 32):
            batch = data[i : i + 32]
            if len(batch) < 2:
                continue

            batch_size = len(batch)
            inputs = torch.LongTensor([seq[0] for seq in batch])
            targets = torch.LongTensor([seq[1] for seq in batch])

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Reinitialize hidden if batch size changed
            if hidden is None or hidden[0].size(1) != batch_size:
                h0 = torch.zeros(model.num_layers, batch_size, model.hidden_dim).to(
                    device
                )
                c0 = torch.zeros(model.num_layers, batch_size, model.hidden_dim).to(
                    device
                )
                hidden = (h0, c0)

            outputs, hidden = model(inputs, hidden)

            # Detach hidden state to truncate backprop through time
            # Without this, BPTT would grow linearly with sequence length
            hidden = (hidden[0].detach(), hidden[1].detach())

            loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
            loss.backward()

            # Log gradient norms to verify LSTM gates prevent vanishing
            # With standard RNN, norm would decay rapidly through time
            # With LSTM, gates allow gradients to flow unchanged
            if log_gradients and total_steps % print_every == 0:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm**0.5
                grad_norms.append(total_norm)
                print(f"Step {total_steps}: Gradient norm = {total_norm:.4f}")

            # Gradient clipping prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

        avg_loss = total_loss / len(data)
        losses.append(avg_loss)

        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return losses, grad_norms


def generate(model, start_str, char_to_idx, idx_to_char, length=200):
    """
    Generate new text given a starting string.

    Uses the trained model to predict and sample next characters.
    Temperature controls randomness: lower = more deterministic.

    Args:
        model: Trained CharLSTM model
        start_str: Initial characters to seed generation
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        length: Number of characters to generate

    Returns:
        Generated text string including the start string
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    generated = start_str
    hidden = None

    # Use last 50 characters as context for next prediction
    for _ in range(length):
        input_seq = torch.LongTensor([[char_to_idx[c] for c in generated[-50:]]]).to(
            device
        )

        output, hidden = model(input_seq, hidden)

        # Sample from probability distribution over next characters
        probs = torch.softmax(output[0, -1], dim=0)
        next_idx = torch.multinomial(probs, 1).item()

        generated += idx_to_char[next_idx]

    return generated


if __name__ == "__main__":
    """
    Main training script.
    
    Steps:
    1. Download Shakespeare corpus (~1MB of text)
    2. Build character vocabulary
    3. Create training sequences
    4. Train LSTM model
    5. Generate sample output
    
    Expected behavior:
    - Initial output: random-looking characters
    - After 1k steps: somewhat coherent word-like patterns
    - After 10k steps: readable Shakespeare-like text
    - After 100k steps: high-quality generated text
    
    Gradient norms should remain stable (not decay to zero),
    demonstrating LSTM's solution to vanishing gradients.
    """
    # Step 1: Get training data
    filepath = download_shakespeare()

    with open(filepath, "r") as f:
        text = f.read()

    print(f"Dataset size: {len(text)} characters")

    # Step 2: Build vocabulary
    chars, char_to_idx, idx_to_char = preprocess_text(text)
    print(f"Vocab size: {len(chars)}")

    # Step 3: Create training sequences
    seq_length = 50
    data = create_sequences(text, char_to_idx, seq_length)
    print(f"Number of sequences: {len(data)}")

    # Step 4: Create model
    model = CharLSTM(
        vocab_size=len(chars), embedding_dim=256, hidden_dim=512, num_layers=2
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step 5: Train and generate (~31k steps with 1 epoch, batch 32)
    print("\nTraining...")
    losses, grad_norms = train(model, data, epochs=1, lr=0.001, print_every=500)

    print("\nSample output:")
    print(generate(model, "ROMEO:", char_to_idx, idx_to_char, length=200))
