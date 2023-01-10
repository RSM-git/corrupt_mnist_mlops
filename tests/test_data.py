import os

import torch
from tests import _PATH_DATA


def load_data(train: bool = True):
    if train:
        return torch.load(os.path.join(_PATH_DATA, "processed", "train.pt"))
    return torch.load(os.path.join(_PATH_DATA, "processed", "test.pt"))

def test_make_dataset():
    assert load_data() is not None


def test_training_len():
    assert len(load_data()["images"]) == 25_000


def test_test_len():
    assert len(load_data(train=False)["images"]) == 5_000


def test_shape():
    assert load_data()["images"].shape == (25_000, 1, 28, 28)


def test_num_samples():
    data = load_data()
    assert len(data["images"]) == len(data["labels"])


def test_normalized():
    data = load_data()["images"]
    assert (abs(torch.mean(data)) < 1e-5) and (abs(torch.std(data) - 1) < 1e-5)
