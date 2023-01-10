import os

import torch
from src.data.dataset import MyDataset
from tests import _PATH_DATA

train = torch.load(os.path.join(_PATH_DATA, "processed", "train.pt"))


def test_dataset():
    dataset = MyDataset(train)
    assert len(dataset) == 25_000
    assert dataset[0][0].shape == (1, 28, 28)
    assert dataset[0][1].shape == ()
