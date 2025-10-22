
import os 
import torch 
from torch.utils.data import Dataset
import numpy as np 


class EllipsesDataset(Dataset):
    def __init__(self, base_path="dataset", transform=None):

        self.file_path = base_path
        self.file_names = os.listdir(self.file_path)

        self.transform = transform

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, IDX):
        sigma = np.load(os.path.join(self.file_path, self.file_names[IDX])) 

        if self.transform:
            sigma = self.transform(sigma)

        return torch.from_numpy(sigma).float()


# Example transform: normalization
class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


if __name__ == "__main__":

    mean = 7.5 
    std = 9.0   
    transform = Normalize(mean, std)

    dataset = EllipsesDataset(base_path="dataset/darcy_flow", transform=transform)

    print("Length: ", len(dataset))

    x = dataset[0]

    print(x.shape)
    print("x.min: ", x.min(), " x.max: ", x.max())
    plot_batch = [dataset[i] for i in range(6)]
    plot_batch = torch.cat(plot_batch, dim=0).unsqueeze(1)
    print(plot_batch.shape)