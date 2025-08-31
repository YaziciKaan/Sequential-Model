import torch
import tqdm

from model import SimpleNN
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()

# Create DataLoaders from MNIST dataset
def get_data_loaders():
    train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=512, shuffle=False)

    return train_dataloader, test_dataloader

# Create Model
def get_model():
    model = SimpleNN()
    return model

# Get Device for Training
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Function
def train(epoch=10):
    model = get_model()
    device = get_device()

    model.to(device)

    train_dataloader, _ = get_data_loaders()

    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    for e in range(epoch):
        pbar = tqdm.tqdm(train_dataloader, desc=f"Epoch {e+1}/{epoch}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {e+1}/{epoch}, Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), "model.pth")


def test():
    model = get_model()
    device = get_device()

    model.to(device)

    _, test_dataloader = get_data_loaders()

    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    corrects = []

    for images, labels in tqdm.tqdm(test_dataloader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        tqdm.tqdm.write(f"Test Loss: {loss.item():.4f}")
        _, preds = torch.max(outputs, dim=1)
        corrects.append((preds == labels).sum().item())

    Accuracy = sum(corrects) / len(test_dataloader.dataset)
    tqdm.tqdm.write(f"Test Accuracy: {Accuracy:.4f}")


if __name__ == "__main__":
    print("Training Model...")
    train()
    print("Testing Model...")
    test()