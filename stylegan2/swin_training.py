import zipfile
import io
import torch
from torch.utils.data import DataLoader, RandomSampler
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
import shutil
from FFHQDataset import FFHQDataset
from training.ms_ssim_l1_loss import MS_SSIM_L1_LOSS
from accelerate import Accelerator 
from SwinDiscrimminator import SwinDiscrimminator


def plot_loss(epochs, train_loss, val_loss, path=None):
    plt.plot(epochs, train_loss, 'b', label='Train Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    if path:
      plt.savefig(path)

def save_snapshot(normalised_original_image, normalised_degraded_image, restored_image, epoch, snapshot_dir):
    original_images = (normalised_original_image.permute(0, 2, 3, 1).detach().cpu().numpy()+1)/2 #[0,1]
    degraded_images = (normalised_degraded_image.permute(0, 2, 3, 1).detach().cpu().numpy()+1)/2 #[0,1]
    restored_images = np.clip((restored_image.permute(0, 2, 3, 1).detach().cpu().numpy()+1)/2, 0, 1) #[0,1]

    num_images = original_images.shape[0]

    columns = [np.vstack((original_images[i], degraded_images[i], restored_images[i])) for i in range(num_images)]
    snapshot_image = np.hstack(columns)

    fig_width = num_images * 3
    fig_height = 9

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.imshow(snapshot_image)
    ax.axis('off')
    snapshot_name = f'iter_{epoch}.png'
    
    plt.savefig(os.path.join(snapshot_dir, snapshot_name), bbox_inches='tight', pad_inches=0)
    plt.close()

def get_losses_from_csv(csv_file):
    with open(csv_file, 'r') as file:
      reader = csv.reader(file)
      next(reader)  # Skip the header row if it exists

      epochs, train_loss, val_loss = zip(*[(int(row[0]), float(row[1]), float(row[2])) for row in reader])
      return epochs, train_loss, val_loss

def write_losses_to_csv(csv_file, epoch, train_loss, val_loss):
    with open(csv_file, 'a') as file:
        if file.tell() == 0:  # Check if file is empty
            file.write("Epoch,Train Loss,Val Loss\n")  # Write header if file is empty
        file.write(f"{epoch},{train_loss:.4f},{val_loss:.4f}\n")

def save_checkpoint(checkpoint_dir, model, epoch, val_loss, max_checkpoints):
    checkpoints = [f for f in os.listdir(checkpoint_dir)]

    # Check if the number of existing checkpoints exceeds the limit
    if len(checkpoints) >= max_checkpoints:
        # Sort the checkpoints by creation time (oldest first)
        checkpoints.sort(key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
        
        # Remove the oldest checkpoints until the number is within the limit
        remove_count = len(checkpoints) - max_checkpoints + 1
        for i in range(remove_count):
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints[i])
            shutil.rmtree(checkpoint_path)

    checkpoint_by_epoch_dir = os.path.join(checkpoint_dir, f'checkpoint_val_loss_{val_loss:.4f}_iter_{epoch}')
    os.makedirs(checkpoint_by_epoch_dir, exist_ok=True)
    encoder_checkpoint_path = os.path.join(checkpoint_by_epoch_dir, 'encoder.pt')
    decoder_checkpoint_path = os.path.join(checkpoint_by_epoch_dir, 'decoder.pt')
    torch.save(model.encoder.state_dict(), encoder_checkpoint_path)
    torch.save(model.decoder.state_dict(), decoder_checkpoint_path)


def save_discrimminator_checkpoint(checkpoint_dir, model, epoch, val_loss, max_checkpoints):
    checkpoints = [f for f in os.listdir(checkpoint_dir)]

    # Check if the number of existing checkpoints exceeds the limit
    if len(checkpoints) >= max_checkpoints:
        # Sort the checkpoints by creation time (oldest first)
        checkpoints.sort(key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
        
        # Remove the oldest checkpoints until the number is within the limit
        remove_count = len(checkpoints) - max_checkpoints + 1
        for i in range(remove_count):
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints[i])
            shutil.rmtree(checkpoint_path)

    checkpoint_by_epoch_dir = os.path.join(checkpoint_dir, f'checkpoint_val_loss_{val_loss:.4f}_iter_{epoch}')
    os.makedirs(checkpoint_by_epoch_dir, exist_ok=True)
    discrimminator_checkpoint_path = os.path.join(checkpoint_by_epoch_dir, 'discrimminator.pt')
    torch.save(model.state_dict(), discrimminator_checkpoint_path)

def load_checkpoint(model, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
      print("Checkpoint folder does not exist")
      return
    encoder_checkpoint = torch.load(os.path.join(checkpoint_dir, 'encoder.pt'))
    model.encoder.load_state_dict(encoder_checkpoint)

    decoder_checkpoint = torch.load(os.path.join(checkpoint_dir, "decoder.pt"))
    model.decoder.load_state_dict(decoder_checkpoint)

def initialize_dataloaders(dataset_dir, validation_split, batch_size, shuffle_dataset, random_seed):
    print("Creating dataset...")
    dataset = FFHQDataset(dataset_dir)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_dataset_size = len(train_indices)
    val_dataset_size = len(val_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                          sampler=train_sampler, num_workers=4)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                          sampler=valid_sampler, num_workers=4)

    return train_loader, val_loader, train_dataset_size, val_dataset_size


def start_training(train_loader, val_loader, train_dataset_size, val_dataset_size, 
          model, criterion, optimizer, degrade, epochs, swin_training_dir, 
          checkpoint_dir, snapshot_dir, csv_file, max_checkpoints, resume_epoch=0):
    
    # accelerator = Accelerator()
    # train_loader, val_loader, model, optimizer = accelerator.prepare(train_loader, val_loader, model, optimizer)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train().to(device)
    best_val_loss = float('inf') 
    print("Start training...")
    for epoch in range(resume_epoch, epochs+resume_epoch):
        running_loss = 0.0
        degrading_times = []
        forward_pass_times = []
        loss_calculation_times = []
        backward_pass_times = []
        weight_update_times = []
        epoch_pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        # Training loop
        for bx, original_image in enumerate(train_loader):
            transformed_result = degrade(original_image)
            normalised_original_image = transformed_result["jpg"].permute(0, 3, 1, 2)/127.5-1
            #print("\nDegrading...")
            start_time = time.time()
            normalised_degraded_image = transformed_result['hint'].permute(0,3,1,2)/127.5-1
            degrading_time = time.time() - start_time
            degrading_times.append(degrading_time)
            #print("Forward pass...")
            start_time = time.time()
            restored_image = model(normalised_degraded_image.to(device))
            forward_time = time.time() - start_time
            forward_pass_times.append(forward_time)
            #print("Calculating Loss...")
            start_time = time.time()
            loss = criterion(normalised_original_image.to(device), restored_image)
            loss_time = time.time() - start_time
            loss_calculation_times.append(loss_time)
            optimizer.zero_grad()
            #print("Backward pass...")
            start_time = time.time()
            loss.backward()
            backward_time = time.time() - start_time
            backward_pass_times.append(backward_time)
            #print("Updating weights...")
            start_time = time.time()
            optimizer.step()
            weight_update_time = time.time() - start_time
            weight_update_times.append(weight_update_time)
            running_loss += loss.item()
            epoch_pbar.update(1)
        
        epoch_loss = running_loss / train_dataset_size
        # Calculate average times
        avg_degrading_time = sum(degrading_times) / len(train_loader)
        avg_forward_pass_time = sum(forward_pass_times) / len(train_loader)
        avg_loss_calculation_time = sum(loss_calculation_times) / len(train_loader)
        avg_backward_pass_time = sum(backward_pass_times) / len(train_loader)
        avg_weight_update_time = sum(weight_update_times) / len(train_loader)

        # Print average times
        print("\nAverage Times:")
        print(f"Degrading time: {avg_degrading_time:.4f} seconds")
        print(f"Forward pass time: {avg_forward_pass_time:.4f} seconds")
        print(f"Loss calculation time: {avg_loss_calculation_time:.4f} seconds")
        print(f"Backward pass time: {avg_backward_pass_time:.4f} seconds")
        print(f"Weight update time: {avg_weight_update_time:.4f} seconds")
        
        # Validation loop
        model.eval() 
        val_loss = 0.0
        with torch.no_grad():
            saved_snapshot = False
            for bx, original_image in enumerate(val_loader):
                transformed_result = degrade(original_image)
                normalised_original_image = transformed_result["jpg"].permute(0, 3, 1, 2)/127.5-1 #[-1,1]
                normalised_degraded_image = transformed_result['hint'].permute(0,3,1,2)/127.5-1 #[-1,1]
                restored_image = model(normalised_degraded_image.to(device)) #[-1,1]
                loss = criterion(normalised_original_image.to(device), restored_image)
                val_loss += loss.item()
                if not saved_snapshot and (epoch == 0 or (epoch+1)%1 == 0 or epoch == epochs-1):
                    save_snapshot(normalised_original_image, normalised_degraded_image, restored_image, epoch+1, snapshot_dir)
                    saved_snapshot = True

        epoch_val_loss = val_loss / val_dataset_size
        model.train()

        write_losses_to_csv(csv_file, epoch, epoch_loss, epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_checkpoint(checkpoint_dir, model, epoch+1, epoch_val_loss, max_checkpoints)
        
        print('[EPOCH] {}/{}\n[TRAIN LOSS] {}\n[VAL LOSS] {}'.format(epoch + 1, epochs, epoch_loss, epoch_val_loss))
        epoch_pbar.close()

def start_adversarial_training(train_loader, val_loader, train_dataset_size, val_dataset_size, 
          generator, discrimminator, generator_criterion, discrimminator_criterion, 
          generator_optimizer, discrimminator_optimizer, degrade, epochs, swin_training_dir, 
          generator_checkpoint_dir, discrimminator_checkpoint_dir, snapshot_dir, generator_csv_file, discrimminator_csv_file, max_checkpoints, real_label, fake_label, resume_epoch=0):
    
    # accelerator = Accelerator()
    # train_loader, val_loader, model, optimizer = accelerator.prepare(train_loader, val_loader, model, optimizer)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator.train().to(device)
    discrimminator.train().to(device)
    best_generator_val_loss = float('inf')
    best_discrimminator_val_loss = float('inf') 
    print("Start training...")
    for epoch in range(resume_epoch, epochs+resume_epoch):
        generator_running_loss = 0.0
        discrimminator_running_loss = 0.0
        epoch_pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        # Training loop
        for bx, original_image in enumerate(train_loader):
            #print("\nDegrading...")
            transformed_result = degrade(original_image)
            normalised_original_image = transformed_result["jpg"].permute(0, 3, 1, 2)/127.5-1
            label = torch.full((normalised_original_image.shape[0],), real_label, dtype=torch.float, device=device)
            discrimminator_output = discrimminator(normalised_original_image.to(device)).view(-1)
            discrimminator_real_img_loss = discrimminator_criterion(discrimminator_output, label)
            discrimminator_optimizer.zero_grad()
            discrimminator_real_img_loss.backward()

            normalised_degraded_image = transformed_result['hint'].permute(0,3,1,2)/127.5-1
            restored_image = generator(normalised_degraded_image.to(device))
            label.fill_(fake_label)
            discrimminator_output = discrimminator(restored_image.detach()).view(-1)
            discrimminator_fake_img_loss = discrimminator_criterion(discrimminator_output, label)
            discrimminator_fake_img_loss.backward()

            discrimminator_loss = discrimminator_fake_img_loss + discrimminator_real_img_loss
            discrimminator_optimizer.step()
            discrimminator_running_loss += discrimminator_loss.item()


            generator_optimizer.zero_grad()
            discrimminator.eval()
            label.fill_(real_label)
            discrimminator_output = discrimminator(restored_image).view(-1)
            generator_loss = 0.2*discrimminator_criterion(discrimminator_output, label) + generator_criterion(normalised_original_image.to(device), restored_image)
            generator_loss.backward()
            generator_optimizer.step()
            discrimminator.train()
            generator_running_loss += generator_loss.item()
            epoch_pbar.update(1)
        
        generator_epoch_loss = generator_running_loss / train_dataset_size
        discrimminator_epoch_loss = discrimminator_running_loss / train_dataset_size

        # Validation loop
        generator.eval()
        discrimminator.eval()
        generator_val_loss = 0.0
        discrimminator_val_loss = 0.0
        with torch.no_grad():
            saved_snapshot = False
            for bx, original_image in enumerate(val_loader):
                transformed_result = degrade(original_image)
                normalised_original_image = transformed_result["jpg"].permute(0, 3, 1, 2)/127.5-1 #[-1,1]
                normalised_degraded_image = transformed_result['hint'].permute(0,3,1,2)/127.5-1 #[-1,1]
                label = torch.full((normalised_original_image.shape[0],), real_label, dtype=torch.float, device=device)
                discrimminator_output = discrimminator(normalised_original_image.to(device)).view(-1)
                discrimminator_real_img_loss = discrimminator_criterion(discrimminator_output, label)

                restored_image = generator(normalised_degraded_image.to(device))
                label.fill_(fake_label)
                discrimminator_output = discrimminator(restored_image.detach()).view(-1)
                discrimminator_fake_img_loss = discrimminator_criterion(discrimminator_output, label)
                
                discrimminator_loss = discrimminator_fake_img_loss + discrimminator_real_img_loss
                discrimminator_val_loss += discrimminator_loss.item()

                label.fill_(real_label)
                discrimminator_output = discrimminator(restored_image).view(-1)
                generator_loss = 0.2*discrimminator_criterion(discrimminator_output, label) + generator_criterion(normalised_original_image.to(device), restored_image)
                generator_val_loss += generator_loss.item()
                if not saved_snapshot and (epoch == 0 or (epoch+1)%1 == 0 or epoch == epochs-1):
                    save_snapshot(normalised_original_image, normalised_degraded_image, restored_image, epoch+1, snapshot_dir)
                    saved_snapshot = True

        generator_epoch_val_loss = generator_val_loss / val_dataset_size
        discrimminator_epoch_val_loss = discrimminator_val_loss / val_dataset_size
        generator.train()
        discrimminator.train()

        write_losses_to_csv(generator_csv_file, epoch, generator_epoch_loss, generator_epoch_val_loss)
        write_losses_to_csv(discrimminator_csv_file, epoch, discrimminator_epoch_loss, discrimminator_epoch_val_loss)

        # if generator_epoch_val_loss < best_generator_val_loss:
        #     best_generator_val_loss = generator_epoch_val_loss
        save_checkpoint(generator_checkpoint_dir, generator, epoch+1, generator_epoch_val_loss, max_checkpoints)

        # if discrimminator_epoch_val_loss < best_discrimminator_val_loss:
        #     best_discrimminator_val_loss = discrimminator_epoch_val_loss
        save_discrimminator_checkpoint(discrimminator_checkpoint_dir, discrimminator, epoch+1, discrimminator_epoch_val_loss, max_checkpoints)
        
        print('[EPOCH] {}/{}\n[GENERATOR TRAIN LOSS] {}\n[ DISCRIMMINATOR TRAIN LOSS] {}\n[GENERATOR VAL LOSS] {}\n[DISCRIMMINATOR VAL LOSS]{}'.format(epoch + 1, 
        epochs, generator_epoch_loss, discrimminator_epoch_loss, generator_epoch_val_loss, discrimminator_epoch_loss))
        epoch_pbar.close()

def main():
    dataset_file_path = "/content/images256x256"
    swin_training_dir = 'swin_training'
    checkpoint_dir = os.path.join(swin_training_dir, 'autoencoder_checkpoints')
    generator_checkpoint_dir = os.path.join(checkpoint_dir, "generator")
    discrimminator_checkpoint_dir = os.path.join(checkpoint_dir, "discrimminator")
    snapshot_dir = os.path.join(swin_training_dir, "swin pretrain snapshot")
    generator_csv_file = os.path.join(swin_training_dir, 'generator_losses.csv')
    discrimminator_csv_file = os.path.join(swin_training_dir, 'discrimminator_losses.csv')
    degradation_config_path = "./degradation/configs/general_deg_realesrgan_train.yaml"
    csv_file = os.path.join(swin_training_dir, 'losses.csv')
    generator_loss_plot_dir = os.path.join(swin_training_dir, "generator_loss.png")
    discrimminator_loss_plot_dir = os.path.join(swin_training_dir, "discrimminator_loss.png")

    max_checkpoints = 5

    use_existing_checkpoint = False
    checkpoint_iter = 0
    checkpoint_val_loss = 0
    model_checkpoint_dir = os.path.join(checkpoint_dir, f'checkpoint_val_loss_{checkpoint_val_loss:.4f}_iter_{checkpoint_iter}')
    
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42
    batch_size = 32

    learning_rate = 0.0005
    betas = (0.5, 0.999)
    epsilon = 1e-8
    epochs = int(10)

    real_label = 1
    fake_label = 0

    degradation_config = OmegaConf.load(degradation_config_path)
    
    degrade = instantiate_from_config(degradation_config.batch_transform)
    train_loader, val_loader, train_dataset_size, val_dataset_size = initialize_dataloaders(dataset_file_path, validation_split, batch_size, shuffle_dataset, random_seed)
    model = SwinAutoEncoder()
    generator = SwinAutoEncoder()
    discrimminator = SwinDiscrimminator()
    criterion = MS_SSIM_L1_LOSS(alpha=0.7)
    generator_criterion = nn.L1Loss()
    discrimminator_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=epsilon)
    generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=betas, eps=epsilon)
    discrimminator_optimizer = optim.Adam(discrimminator.parameters(), lr=learning_rate, betas=betas, eps=epsilon)
    if use_existing_checkpoint:
      load_checkpoint(model, model_checkpoint_dir)

    generator_total_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    discrimminator_total_params = sum(p.numel() for p in discrimminator.parameters() if p.requires_grad)

    print("Training set size:", train_dataset_size)
    print("Validation set size:", val_dataset_size)
    print("Number of batches in the training loader:", len(train_loader))
    print("Number of batches in the validation loader:", len(val_loader))
    print(f"Total number of parameters in generator: {generator_total_params}")
    print(f"Total number of parameters in discrimminator: {discrimminator_total_params}")

    if not os.path.exists(generator_checkpoint_dir):
        os.makedirs(generator_checkpoint_dir)

    if not os.path.exists(discrimminator_checkpoint_dir):
        os.makedirs(discrimminator_checkpoint_dir)

    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    # start_training(train_loader, val_loader, train_dataset_size, 
    #         val_dataset_size, model, criterion, optimizer, 
    #         degrade, epochs, swin_training_dir, checkpoint_dir, 
    #         snapshot_dir, csv_file, max_checkpoints, resume_epoch=checkpoint_iter)

    start_adversarial_training(train_loader, val_loader, train_dataset_size,
                  val_dataset_size, generator, discrimminator, 
                  generator_criterion, discrimminator_criterion,
                  generator_optimizer, discrimminator_optimizer,
                  degrade, epochs, swin_training_dir, generator_checkpoint_dir, 
                  discrimminator_checkpoint_dir, snapshot_dir, 
                  generator_csv_file, discrimminator_csv_file, 
                  max_checkpoints, real_label, fake_label)

    epochs, generator_train_loss, generator_val_loss = get_losses_from_csv(generator_csv_file)
    plot_loss(epochs, generator_train_loss, generator_val_loss, generator_loss_plot_dir)
    epochs, discrimminator_train_loss, discrimminator_val_loss = get_losses_from_csv(discrimminator_csv_file)
    plot_loss(epochs, discrimminator_train_loss, discrimminator_val_loss, discrimminator_loss_plot_dir)

if __name__ == "__main__":
    main()