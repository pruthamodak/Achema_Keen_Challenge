from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from random import shuffle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class KeenDataloader():
    def __init__(self, root, is_training=False, transforms=None):
        self.transforms = transforms
        self.is_training = is_training
        self.images = self.get_paths(root)
        self.labels = {"Fluten" : 0, "Normalzustand" : 1}
    
    def get_paths(self, root):
        img_format = ['.jpg', '.png']

        dirs = [x[0] for x in os.walk(root, followlinks=True) if not x[0].startswith('.')]
        datasets = []
        for fdir in dirs:
            for el in os.listdir(fdir):
                if os.path.isfile(os.path.join(fdir, el)) and \
                not el.startswith('.') and \
                any([el.endswith(ext) for ext in img_format]):
                    datasets.append(os.path.join(fdir,el))
        shuffle(datasets)
        return datasets

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        image = Image.open(image)
        image = torch.from_numpy(np.array(image, dtype=np.float32).transpose((2,0,1)))
        if self.is_training:
            if self.transforms is None:
                #
                self.transforms = transforms.Compose([
                                                      transforms.Normalize((0.5021, 0.4780, 0.4723), (0.3467, 0.3395, 0.3367)),
                                                      transforms.RandomCrop((256, 256)),
                                                      transforms.RandomVerticalFlip(0.5),
                                                      transforms.RandomHorizontalFlip(0.5),
                                                      ])    
        else:
            self.transforms = transforms.Resize((256, 256))
        image = self.transforms(image)
        return {'image' : image, 'label' : torch.tensor(self.labels[os.path.basename(os.path.dirname(self.images[index]))], dtype=torch.float32)}         

def get_mean_and_std(dataloader, batch_size):
    channels_sum, channels_sum_squared, num_batches = 0, 0, 0
    for data in tqdm(dataloader):
        image = data['image']
        channels_sum += torch.mean(image, dim=[0,2,3])
        channels_sum_squared += torch.mean(image**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_sum_squared / num_batches - mean**2)**0.5
    logging.info(f"mean : {mean}, std : {std}, mean_1 : {channels_sum / (num_batches*batch_size)}, std_1 : {(channels_sum_squared / (num_batches*batch_size) - mean**2)**0.5} ")
    return mean, std

if __name__ == '__main__':
    from tqdm import tqdm
    import torchvision.transforms.functional as F
    import logging
    logging.basicConfig(level=logging.DEBUG,
                    filename=os.path.join("C:\\Users\\Karthik\\Desktop\\checkpoints", 'mean.log'),
                    format='%(asctime)s %(message)s')
    datasets = KeenDataloader("C:\\Users\\Karthik\\Documents\\KEEN_DATA\\Training", is_training=True)
    tkwargs = {'batch_size': 1,
               'num_workers': 1,
               'pin_memory': True, 'drop_last': True}

    #train_loader = DataLoader(datasets, **tkwargs)
    #mean, std = get_mean_and_std(train_loader, 16)
    train_loader = DataLoader(datasets, **tkwargs)
    for i, sample in enumerate(train_loader):
        print(torch.unique(sample['image'][0]), sample['label'])
        plt.imshow(sample['image'][0].numpy().transpose((1,2,0)))
        plt.savefig(os.path.join("C:\\Users\\Karthik\\Desktop\\checkpoints", f"{i}.png"))
        if (i==10):
            break
