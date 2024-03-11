import zipfile
import io
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from collections import defaultdict
from datetime import timedelta
import time
import torch.optim as optim
from AutoEncoder import SwinAutoEncoder
import torch.nn as nn
from degradation.utils.common import instantiate_from_config, load_state_dict
from omegaconf import OmegaConf
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import csv
from torch.utils.data.sampler import SubsetRandomSampler


class FFHQDataset(Dataset):
    def __init__(self, extract_dir, transform=None):
        self.extract_dir = extract_dir
        self.transform = transform

        self.image_files = [
            file_name
            for file_name in os.listdir(extract_dir)
            if file_name.endswith(".png")
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.extract_dir, self.image_files[idx])
        image = self.load_image(image_path)

        return image

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return ToTensor()(image)

def plot_loss(epochs, train_loss, val_loss):
    plt.plot(epochs, train_loss, 'b', label='Train Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Specify the path to the zip file
zip_file_path = "/content/drive/MyDrive/dataset/ffhq/images256x256"

# Define the transform to be applied to the images (if needed)
config = OmegaConf.load("./degradation/configs/general_deg_realesrgan_train.yaml")
transform = instantiate_from_config(config.batch_transform)
print("Creating dataset...")
# Create an instance of the dataset
dataset = FFHQDataset(zip_file_path)
print("Created datset variable")

validation_split = .2
shuffle_dataset = True
random_seed= 42
batch_size = 32

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_dataset_size = len(train_indices)
val_dataset_size = len(val_indices)
# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
print("Creating dataloder")
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
print("Finished creating dataloder")


# Access the data loaders
print("Training set size:", train_dataset_size)
print("Validation set size:", val_dataset_size)
print("Number of batches in the training loader:", len(train_loader))
print("Number of batches in the validation loader:", len(val_loader))

# Iterate through the data loaders

learning_rate = 0.001
betas = (0.9, 0.999)
epsilon = 1e-8
epochs = 10000

metrics = defaultdict(list)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SwinAutoEncoder()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=epsilon)

swin_training_dir = 'swin_training/'

checkpoint_dir = swin_training_dir + 'autoencoder_checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

snapshot_dir = swin_training_dir + "swin pretrain snapshot"
if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)

model.train()
start = time.time()
best_val_loss = float('inf') 
print("Start training...")
for epoch in range(epochs):
    running_loss = 0.0
    
    # Training loop
    for bx, original_image in tqdm(enumerate(train_loader)):
        normalised_original_image = original_image / 127.5 - 1
        normalised_degraded_image = transform(original_image).permute(0,3,1,2)/127.5-1
        restored_image = model(normalised_degraded_image.to(device))
        loss = criterion(normalised_original_image.to(device), restored_image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / train_dataset_size
    metrics['train_loss'].append(epoch_loss)
    
    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation during validation
        saved_snapshot = False
        for bx, original_image in val_loader:
            normalised_original_image = original_image / 127.5 - 1
            normalised_degraded_image = transform(original_image).permute(0,3,1,2)/127.5-1
            restored_image = model(normalised_degraded_image.to(device))
            loss = criterion(normalised_original_image.to(device), restored_image)
            val_loss += loss.item()
            if epoch%50==0 and not saved_snapshot:
                # Convert torch tensors to numpy arrays
                original_images = (normalised_original_image.permute(0, 2, 3, 1).detach().cpu().numpy()+1)*127.5
                degraded_images = (normalised_degraded_image.permute(0, 2, 3, 1).detach().cpu().numpy()+1)*127.5
                restored_images = (restored_image.permute(0, 2, 3, 1).detach().cpu().numpy()+1)*127.5

                # Configure the plot grid
                num_images = original_images.shape[0]  # Number of images in the batch
                grid_size = int(np.ceil(np.sqrt(num_images)))  # Size of the grid (square root of num_images, rounded up)

                # Create a figure and axis for the plot
                fig, axs = plt.subplots(grid_size, grid_size)

                # Iterate over the images in the batch and plot them in the grid
                for i in range(num_images):
                    row = i // grid_size
                    col = i % grid_size
                    axs[row, col].imshow(np.hstack((original_images[i], degraded_images[i], restored_images[i])))
                    axs[row, col].axis('off')

                snapshot_name = f'iter_{epoch}.png'
                # Save the plot as an image
                plt.savefig(snapshot_dir+'/'+snapshot_name)
                saved_snapshot = True

    epoch_val_loss = val_loss / val_dataset_size
    metrics['val_loss'].append(epoch_val_loss)
    model.train()  # Set the model back to training mode

    with open(swin_training_dir+'losses.csv', 'a') as file:
        if file.tell() == 0:  # Check if file is empty
            file.write("Epoch,Train Loss,Val Loss\n")  # Write header if file is empty
        file.write(f"{epoch},{epoch_loss:.4f},{epoch_val_loss:.4f}\n")

    # Check if the current model has the lowest validation loss
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        checkpoint_by_epoch_dir = f'checkpoint_val_loss_{epoch_val_loss:.4f}_iter_{epoch}'
        os.makedirs(checkpoint_dir+'/'+checkpoint_by_epoch_dir, exist_ok=True)
        encoder_checkpoint_path = os.path.join(checkpoint_dir+'/'+checkpoint_by_epoch_dir, 'encoder.pt')
        decoder_checkpoint_path = os.path.join(checkpoint_dir+'/'+checkpoint_by_epoch_dir, 'decoder.pt')
        torch.save(model.encoder.state_dict(), encoder_checkpoint_path)
        torch.save(model.decoder.state_dict(), decoder_checkpoint_path) # Save the model
    
    print('[EPOCH] {}/{}\n[TRAIN LOSS] {}\n[VAL LOSS] {}'.format(epoch + 1, epochs, epoch_loss, epoch_val_loss))
    

# Assuming you have a CSV file with epoch, train loss, and val loss columns
csv_file = swin_training_dir+'losses.csv'

epochs = []
train_loss = []
val_loss = []

with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row if it exists
    
    for row in reader:
        epochs.append(int(row[0]))
        train_loss.append(float(row[1]))
        val_loss.append(float(row[2]))

plot_loss(epochs, train_loss, val_loss)