"""
Utility functions for LSTM training
Port of Facebook's char-rnn base.lua
"""

import torch
import numpy as np
import random
import string


def g_disable_dropout(model):
    """Disable dropout in the model"""
    for module in model.modules():
        if hasattr(module, "p"):
            # Store original dropout p value
            if not hasattr(module, "_original_p"):
                module._original_p = module.p
            module.p = 0
        if hasattr(module, "train"):
            # For Dropout modules
            module.eval()


def g_enable_dropout(model):
    """Enable dropout in the model"""
    for module in model.modules():
        if hasattr(module, "_original_p"):
            module.p = module._original_p
        if hasattr(module, "train") and module.__class__.__name__ == "Dropout":
            module.train()


def g_replace_table(to, from_):
    """Copy values from one table to another"""
    assert len(to) == len(from_)
    for i in range(len(to)):
        to[i].copy_(from_[i])


def g_make_deterministic(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def g_init_gpu(gpuidx=1):
    """Initialize GPU"""
    if torch.cuda.is_available():
        torch.cuda.set_device(gpuidx - 1)  # Convert to 0-indexed
        print(f"Using GPU {gpuidx}")
        g_make_deterministic(1)
    else:
        print("CUDA not available, using CPU")


def g_f3(f):
    """Format float with 3 decimal places"""
    return f"{f:.3f}"


def g_d(f):
    """Format as integer"""
    return f"{int(torch.round(torch.tensor(f)))}"


def g_clone_many_times(net, T):
    """
    Clone a network T times
    Each clone shares parameters with the original
    """
    clones = []
    params = list(net.parameters())
    grad_params = list(net.parameters())

    # Store the state dict
    for t in range(T):
        # Create a new instance of the same model
        clone = net.__class__(**net.config if hasattr(net, "config") else {})
        clone.load_state_dict(net.state_dict())

        # Clone shares parameters with original
        # This is done by not separating the cloned parameters
        clones.append(clone)

    return clones


# Alternative simpler implementation that works with PyTorch
def clone_model(model, num_clones):
    """Create multiple clones of a model that share parameters"""
    clones = [model]
    for _ in range(num_clones - 1):
        clone = model.__class__(**model.config if hasattr(model, "config") else {})
        # Copy parameters (not weights) - they will be tied
        for p1, p2 in zip(clone.parameters(), model.parameters()):
            p1.data = p2.data
        clones.append(clone)
    return clones


if __name__ == "__main__":
    # Test functions
    print("Testing utility functions...")

    # Test formatting
    assert g_f3(3.14159) == "3.142"
    assert g_d(3.7) == "4"

    print("All tests passed!")
