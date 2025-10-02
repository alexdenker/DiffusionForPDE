
import os 
import torch 
from torch.utils.data import Dataset
import numpy as np 


class EllipsesDataset(Dataset):
    def __init__(self, base_path="dataset"):

        self.file_path = base_path
        self.file_names = os.listdir(self.file_path)

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, IDX):
        sigma = np.load(os.path.join(self.file_path, self.file_names[IDX])) 

        return torch.from_numpy(sigma).float()


if __name__ == "__main__":

    dataset = EllipsesDataset(base_path="dataset/mesh_dg0")

    print("Length: ", len(dataset))

    x = dataset[0]

    print(x.shape)

    plot_batch = [dataset[i] for i in range(6)]
    plot_batch = torch.cat(plot_batch, dim=0).unsqueeze(1)
    print(plot_batch.shape)