import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import random_split

from model import DiffusionSimple

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Data parameters
RESOLUTION = 64
BATCH_SIZE = 64

# Transforms image data
transform = transforms.Compose([
    transforms.Resize((RESOLUTION, RESOLUTION)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def save_model(model, filepath="model.pth"):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

    import os


def load_model(model, filepath="model.pth"):
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath))
        model.eval()
        print(f"Model loaded from {filepath}")
    else:
        print(f"No saved model found at {filepath}")



# START BY GETTING MEAN AND STD OF DATA SET. NEED IT FOR IMAGE HANDLING
torch.set_default_device("cpu")
data = torchvision.datasets.LFWPeople(root='./data', download=True, transform=transform)

def torch_to_pil(img):

    reverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-1,-1,-1], std=[2,2,2]),  # Reverse normalization
        transforms.ToPILImage()
    ])
    return reverse_transform(img)

def pil_to_torch(img):
    # Apply the custom normalization after converting to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(img)

def sample_and_plot(model, n, epoch):
    # Sample n images and the intermediate denoising steps at specified time steps
    img, steps = model.sample_img(n=n)  # steps contains images at t=200, t=150, t=100, t=50

    print(f"The min and max values in sampled image: {img[0].min()}, {img[0].max()}")

    # Number of columns (time steps + final output)
    cols = len(steps) + 1  # len(steps) is 4 for t=200, t=150, t=100, t=50 + final output

    col_labels = ["t=200", "t=150", "t=100", "t=50", "output"]
    fig, axes = plt.subplots(n, cols, figsize=(4*cols, 4*n))  # Adjust figsize for clarity

    # For each sampled image (row)
    for row in range(n):
        # Plot the intermediate steps and final output for each image
        for col in range(cols):
            if col == cols - 1:
                # Last column: plot the final output image (x_0)
                axes[row][col].imshow(torch_to_pil(img[row]))
                if row == 0:
                    axes[0][col].set_xlabel("output")
            else:
                # Plot the images from the denoising process at time steps t=200, t=150, t=100, t=50
                axes[row][col].imshow(torch_to_pil(steps[col][row]))  # steps[col] gives the correct t-step image for the row
                if row == 0:
                    axes[0][col].set_xlabel(col_labels[col])

            axes[row][col].axis('off')  # Hide the axes for cleaner display

    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust spacing between images
    plt.savefig(f"artefacts/samples_{epoch}.png")


# DATA HANDLING
train_size = int(0.4 * len(data))
test_size = len(data) - train_size
train_dataset, test_dataset = random_split(data, [train_size, test_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,)

# View data
data_batch = next(iter(train_dataloader))

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    pil_img = torch_to_pil(data_batch[0][i])
    ax.imshow(np.array(pil_img))
    ax.axis('off')  # Turn off axis labels for cleaner look
plt.tight_layout()
plt.show()

# torch.cuda.empty_cache()
ds = DiffusionSimple(device=device, resolution=RESOLUTION)
load_model(ds, "model.pth")
ds.to(device)
optimiser = Adam(ds.parameters(), 1e-3)

epoch = 0
epochs = 20
loss_curve = []

while epoch < epochs:

    epoch_loss = 0
    iterations = 0
    print("epoch", epoch)
    for y_0, _ in train_dataloader:

        optimiser.zero_grad()
        y_0 = y_0.to(device)

        t = torch.randint(0, ds.T, (len(y_0),)).long().to(device)

        y_t, noise = ds.sample_t(y_0, t)
        y_t.to(device)

        noise_pred = ds.network(y_t, t)

        # loss = F.l1_loss(noise, noise_pred)
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        optimiser.step()

        epoch_loss+=loss.item()
        iterations+=1

        if iterations%10 == 0:
            print("iteration: ", iterations, " Loss: ", loss.item())
            print(f"Allocated Memory: {torch.mps.current_allocated_memory() / (1024**2):.2f} MB")


    avg_loss = epoch_loss/iterations
    print(epoch, "epoch loss: ", avg_loss)
    loss_curve.append(avg_loss)

    # plot results of last epoch
    sample_and_plot(ds, 4, epoch)
    save_model(ds, "model.pth")
    epoch+=1

sample_and_plot(ds, 2, epoch=epochs)




