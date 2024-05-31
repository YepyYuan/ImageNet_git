from __future__ import print_function, division
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import transforms
import os
import random


class BirdDataset(Dataset):
    def __init__(self, image_txt, image_dir, transform=None):
        self.image_list = []
        self.label_list = []
        # self.is_train_list = []
        self.image_dir = image_dir
        self.transform = transform
        self.num_classes = 0
        # self.training = training

        with open(image_txt, 'r') as f:
            line = f.readline()
            while line:
                img_name = line.strip().split(' ')[1]
                label = int(img_name.split('.')[0]) - 1
                self.image_list.append(img_name)
                self.label_list.append(label)
                line = f.readline()

        self.num_classes = max(self.label_list) + 1
        
    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label = self.label_list[idx]
        img_name = os.path.join(self.image_dir,img_name)
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label



# def test_dataset():

#     image_dir = './data/CUB_200_2011/CUB_200_2011/images'
#     image_txt = './data/CUB_200_2011/CUB_200_2011/images.txt'

#     rgb_mean = [0.5,0.5,0.5]
#     rgb_std = [0.5,0.5,0.5]

#     transform_val = transforms.Compose([
#         transforms.Resize((299,299)),
#         transforms.ToTensor(),
#         transforms.Normalize(rgb_mean, rgb_std),
#     ])
#     BirdData = BirdDataset(image_txt, image_dir, transform_val, training= True)
#     print(BirdData.num_classes)
#     dataloader = DataLoader(BirdData, batch_size=16, shuffle=True)
#     for data in dataloader:
#         images,labels = data
#         print(images.size(),labels.size(),labels)


# if __name__=='__main__':
#     test_dataset()