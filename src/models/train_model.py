import click
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from model import MyAwesomeModel


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, idx):
        image = self.data["images"][idx]
        label = self.data["labels"][idx]

        return image, label


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--epochs", default=5, help='number of epochs to train for')
def train(lr, epochs):
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    train = MyDataset(torch.load("data/processed/train.pt"))
    trainloader = DataLoader(train, batch_size=64, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        torch.save(model.state_dict(), "models/model.pt")

    plt.plot(losses)
    plt.xlabel("Training steps")
    plt.ylabel("Training Loss")
    plt.savefig("reports/loss.png")

cli.add_command(train)

if __name__ == "__main__":
    cli()