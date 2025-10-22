from torch.utils.data import Dataset, DataLoader
import torch

class MoleculeDataset(Dataset):
    def __init__(self, path):
        print(f"Loading dataset from {path}")
        print('#' * 50)
        self.data = torch.load(path, weights_only=False)
        print(f"Loaded {len(self.data)} molecules.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]