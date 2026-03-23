# Character-level LSTM Implementation

## Original LSTM Paper Key Insights

### The Vanishing Gradient Problem
Traditional RNNs suffer from vanishing gradients when learning long-term dependencies:
- Error signals exponentially decay as they propagate backward through time
- Cannot learn dependencies beyond 5-10 time steps reliably

### LSTM Solution: Gated Memory Cells
The 1997 paper introduced memory cells with **constant error carousel**:

1. **Memory Cell**: Maintains information over long periods through linear self-recurrence
2. **Input Gate**: Controls what new information enters the cell  
3. **Output Gate**: Controls what information leaves the cell

Key insight: When gates are open (value ≈ 1), gradients flow unchanged through time = no vanishing!

## Char-level LSTM Implementation

### Data Preprocessing
For character-level models:
1. Create character vocabulary from text corpus
2. Map each character to a unique integer (one-hot encoding)
3. Create sequences: input = characters 0 to n-1, target = characters 1 to n

### Model Architecture
```
Input (char index) → Embedding → LSTM → Linear → Output (probs for next char)
```

### Training Steps
1. Load text (Shakespeare corpus)
2. Create character-to-index mapping
3. Build training sequences
4. Train LSTM to predict next character
5. Generate samples at 1k, 10k, 100k steps
6. Log gradient norms to verify no vanishing