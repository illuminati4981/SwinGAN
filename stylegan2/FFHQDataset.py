from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
from tqdm import tqdm


class FFHQDataset(Dataset):
    def __init__(self, extract_dir, transform=None):
        self.extract_dir = extract_dir
        self.transform = transform
        self.to_tensor = ToTensor()

        self.images = [
            self.load_image(os.path.join(self.extract_dir, file_name))
            for file_name in tqdm(os.listdir(extract_dir)[:3200])
            # if file_name.endswith(".png")
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # image_path = os.path.join(self.extract_dir, self.image_files[idx])
        # image = self.load_image(image_path)

        return self.images[idx]

    def load_image(self, image_path):
        #print("opening image from dir")
        image = read_image(image_path)/255
        #print("Converting to tensor")
        return image
