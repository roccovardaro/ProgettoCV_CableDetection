import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.v2 as transforms
from utils import generate_mask_tensor_from_coco_file
from torchvision import tv_tensors

class CableTrainDataset(Dataset):
    def __init__(self, img_dir, image_json_path, size=(700,700),transform=None):

        self.img_dir = img_dir
        self.size = size
        self.transform = transform

        with open(image_json_path, "r") as f:
            self.coco_data = json.load(f)

        # Costruisci mappa file_name -> info immagine
        self.img_info_map = {img['file_name']: img for img in self.coco_data['images']}
        all_files = set(os.listdir(img_dir))
        self.valid_filenames = [fname for fname in self.img_info_map.keys() if fname in all_files]

        # Trasformazioni immagini
        self.img_transform = transforms.Compose([
            transforms.ToDtype(torch.float32, scale=True),  # [0,1]
            transforms.Resize(size),
        ])

    def __len__(self):
        return len(self.valid_filenames)

    def __getitem__(self, idx):
        file_name = self.valid_filenames[idx]
        img_info = self.img_info_map[file_name]
        image_id = img_info['id']

        # Carica immagine
        img_path = os.path.join(self.img_dir, file_name)
        image = read_image(img_path)  # [C,H,W], uint8
        image = self.img_transform(image)  # [C,H,W], float32

        # Maschera
        mask = generate_mask_tensor_from_coco_file(self.coco_data, image_id, size=self.size)  # [H,W], Long

        
        if self.transform:
            img_wrapped = tv_tensors.Image(image)
            mask_wrapped = tv_tensors.Mask(mask)
            
            image, mask = self.transform(img_wrapped, mask_wrapped)


        return image, mask, image_id

