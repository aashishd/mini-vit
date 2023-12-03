# %%
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

from transformer import MiniVit

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
print(DEVICE)
# %% load the MNIST dataset
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
# %%
print(len(train_data))
len(test_data)
# %%
sin, sout = next(iter(train_data))
print(sin.shape)
print(sout)
ToPILImage()(sin).show()
# %% show the first batch as grid
from torchvision.utils import make_grid

grid_data = make_grid(next(iter(train_loader))[0])
plt.imshow(grid_data.permute(-2, -1, 0).numpy())

# %% training one epoch function
def training_loop(epoch, data_loader, model, optimizer, loss_fn):
    # model.train(True)
    losses = []
    for i, (inp, oup) in enumerate(data_loader):
        # process 1 batch
        optimizer.zero_grad()
        if DEVICE != 'cpu':
            inp = inp.to(DEVICE)
            oup = oup.to(DEVICE)
        
        logits = model(inp)
        preds = nn.functional.softmax(logits, dim=-1)
        loss = loss_fn(preds, oup)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if i % 100 == 0 and i > 0:
            gplosses = losses[-100:]
            print(f"Epoch: {epoch}, Iteration: {i}, Mean Loss: {sum(gplosses)/len(gplosses)}")
    epoch_loss = sum(losses) / len(losses)
    return epoch_loss

def val_loop(data_loader, model, loss_fn):
    # model.eval()
    losses = []
    for _, (inp, oup) in enumerate(data_loader):
        with torch.no_grad():
            if DEVICE != 'cpu':
                inp = inp.to(DEVICE)
                oup = oup.to(DEVICE)
            
            logits = model(inp)
            preds = nn.functional.softmax(logits, dim=-1)
            loss = loss_fn(preds, oup)
            
            losses.append(loss.item())
    return sum(losses) / len(losses)
    
    
# %% train the model
model = MiniVit(input_dims=16, hidden_dims=8, num_heads=2, num_layers=2, output_classes=10, patch_size=4, img_dims=(28, 28))
if DEVICE != 'cpu':
    model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(20):
    train_loss = training_loop(epoch, train_loader, model, optimizer, loss_fn)
    val_loss = val_loop(test_loader, model, loss_fn)
    print(f"Epoch: {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}")
# %%
