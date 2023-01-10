import torch
import random
from src.models import model

model = model.MyAwesomeModel()

def test_model():
    input = torch.rand((1, 1, 28, 28))
    assert model(input).shape == (1, 10)

def test_probs():
    batch = random.randint(1, 64)
    input = torch.rand((batch, 1, 28, 28))
    assert torch.nn.functional.softmax(model(input), dim=1).shape == (batch, 10)
    assert torch.all(torch.abs(torch.nn.functional.softmax(model(input), dim=1).sum(dim=1) - torch.ones(batch)) < 1e-5)