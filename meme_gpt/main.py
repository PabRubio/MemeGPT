import torch
import tiktoken

from model import GPTModel
from dataset import create_dataloader
from train_utils import train_model
from utils import plot_losses


# Set seed for reproducibility
torch.manual_seed(123)

# GPT Config
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# Load text data
with open("memes.txt", "r", encoding="utf-8") as file:
    text_data = file.read()

# Data split and data loader setup
train_ratio = 0.80
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader(
    train_data, batch_size=4, max_length=GPT_CONFIG_124M["context_length"], stride=GPT_CONFIG_124M["context_length"])
val_loader = create_dataloader(
    val_data, batch_size=4, max_length=GPT_CONFIG_124M["context_length"], stride=GPT_CONFIG_124M["context_length"])

# Model setup
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

# Optimizer setup
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

# Training setup
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model(
    model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs,
    eval_freq=10, eval_iter=10, tokenizer=tiktoken.get_encoding("gpt2")
)

# Save the model
torch.save(model.state_dict(), "model.pth")

# Plot the losses
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)