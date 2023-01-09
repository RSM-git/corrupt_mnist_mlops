from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, idx):
        image = self.data["images"][idx]
        label = self.data["labels"][idx]

        return image, label