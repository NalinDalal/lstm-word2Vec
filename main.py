"""
LSTM Language Model (PyTorch port of Facebook's char-rnn)
Based on: https://github.com/karpathy/char-rnn
"""

import torch
import torch.nn as nn
import numpy as np


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(f"{path}/train.txt")
        self.valid = self.tokenize(f"{path}/valid.txt")
        self.test = self.tokenize(f"{path}/test.txt")

    def tokenize(self, path):
        """Tokenize file into words"""
        tokens = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                words = line.split() + ["<eos>"]
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        ids = torch.LongTensor(tokens)
        with open(path, "r", encoding="utf-8") as f:
            token = 0
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # LSTM
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output layer
        self.dropout_layer = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-init_range, init_range)

    def forward(self, input, hidden):
        """
        input: (batch_size, seq_len)
        hidden: (h0, c0) where each is (num_layers, batch_size, hidden_size)

        Returns:
        output: (batch_size * seq_len, vocab_size)
        hidden: (h0, c0)
        """
        # Embedding: (batch_size, seq_len) -> (batch_size, seq_len, embed_size)
        emb = self.embedding(input)

        # LSTM
        output, hidden = self.lstm(emb, hidden)

        # Reshape for linear layer
        output = output.contiguous().view(
            output.size(0) * output.size(1), output.size(2)
        )

        # Dropout and linear
        output = self.dropout_layer(output)
        output = self.linear(output)

        return output, hidden

    def init_hidden(self, batch_size, device):
        """Initialize hidden states to zeros"""
        weight = next(self.parameters()).data

        h = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
        c = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()

        return (h, c)


class RNNLM:
    def __init__(self, args, corpus):
        self.args = args
        self.corpus = corpus
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model
        self.model = LSTMModel(
            len(self.corpus.dictionary),
            args.embed_size,
            args.hidden_size,
            args.num_layers,
            args.dropout,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=args.learning_rate, momentum=args.momentum
        )

        self.step = 0
        self.epoch = 0

    def repackage_hidden(self, h):
        """Detach hidden states from computation graph"""
        if isinstance(h, torch.Tensor):
            return h.detach()
        return tuple(self.repackage_hidden(v) for v in h)

    def train_epoch(self, train_data, batch_size, seq_length):
        """Train for one epoch"""
        self.model.train()

        hidden = self.model.init_hidden(batch_size, self.device)

        data = train_data
        num_batches = (len(data) - 1) // (batch_size * seq_length)

        total_loss = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size * seq_length
            seqs = []
            targets = []

            for b in range(batch_size):
                seq_start = start_idx + b * seq_length
                seq = data[seq_start : seq_start + seq_length]
                target = data[seq_start + 1 : seq_start + seq_length + 1]

                seqs.append(seq)
                targets.append(target)

            input = torch.stack(seqs, dim=0).to(self.device)
            target = torch.stack(targets, dim=0).to(self.device).contiguous().view(-1)

            hidden = self.repackage_hidden(hidden)

            self.optimizer.zero_grad()
            output, hidden = self.model(input, hidden)

            loss = self.criterion(output, target)
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.max_grad_norm
            )

            self.optimizer.step()

            self.step += 1

            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")

        return total_loss / num_batches

    def evaluate(self, data, batch_size, seq_length):
        """Evaluate on data"""
        self.model.eval()

        hidden = self.model.init_hidden(batch_size, self.device)

        data = data
        num_batches = (len(data) - 1) // (batch_size * seq_length)

        total_loss = 0
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size * seq_length
                seqs = []
                targets = []

                for b in range(batch_size):
                    seq_start = start_idx + b * seq_length
                    seq = data[seq_start : seq_start + seq_length]
                    target = data[seq_start + 1 : seq_start + seq_length + 1]

                    seqs.append(seq)
                    targets.append(target)

                input = torch.stack(seqs, dim=0).to(self.device)
                target = (
                    torch.stack(targets, dim=0).to(self.device).contiguous().view(-1)
                )

                output, hidden = self.model(input, hidden)
                loss = self.criterion(output, target)
                total_loss += loss.item()

        return total_loss / num_batches

    def train(self):
        """Main training loop"""
        args = self.args

        batch_size = args.batch_size
        seq_length = args.seq_length
        epochs = args.epochs

        train_data = self.corpus.train
        val_data = self.corpus.valid
        test_data = self.corpus.test

        best_test_loss = float("inf")

        for epoch in range(epochs):
            self.epoch = epoch

            # Train
            train_loss = self.train_epoch(train_data, batch_size, seq_length)
            train_perplexity = np.exp(train_loss)

            # Validate
            val_loss = self.evaluate(val_data, batch_size, seq_length)
            val_perplexity = np.exp(val_loss)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(
                f"  Train Loss: {train_loss:.4f}, Train Perplexity: {train_perplexity:.2f}"
            )
            print(f"  Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")

            # Learning rate decay
            if epoch >= args.max_epoch:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] /= args.decay

            # Test
            test_loss = self.evaluate(test_data, batch_size, seq_length)
            test_perplexity = np.exp(test_loss)
            print(
                f"  Test Loss: {test_loss:.4f}, Test Perplexity: {test_perplexity:.2f}"
            )

            if test_loss < best_test_loss:
                best_test_loss = test_loss

        print(f"Best test perplexity: {np.exp(best_test_loss):.2f}")


import argparse


def main():
    parser = argparse.ArgumentParser(description="LSTM Language Model")
    parser.add_argument(
        "--data",
        type=str,
        default="./data/pennchar",
        help="location of the data corpus",
    )
    parser.add_argument(
        "--embed_size", type=int, default=256, help="size of word embeddings"
    )
    parser.add_argument("--hidden_size", type=int, default=512, help="hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size")
    parser.add_argument("--seq_length", type=int, default=20, help="sequence length")
    parser.add_argument("--epochs", type=int, default=13, help="number of epochs")
    parser.add_argument(
        "--max_epoch", type=int, default=4, help="epoch after which lr decay starts"
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout")
    parser.add_argument(
        "--learning_rate", type=float, default=1.0, help="learning rate"
    )
    parser.add_argument("--momentum", type=float, default=0.0, help="momentum")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=5.0,
        help="max gradient norm for clipping",
    )
    parser.add_argument("--decay", type=float, default=2.0, help="learning rate decay")

    args = parser.parse_args()

    # Load corpus
    print("Loading corpus...")
    corpus = Corpus(args.data)
    print(f"Vocab size: {len(corpus.dictionary)}")

    # Train
    model = RNNLM(args, corpus)
    model.train()


if __name__ == "__main__":
    main()
