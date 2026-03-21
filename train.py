warnings.filterwarnings('ignore')
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse
from model import MovieposterNet
import warnings


 
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"[INFO] Utilisation du device : {device}")

# Fonction pour tester le modèle
def test(net, loader):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_hat = net(x).argmax(1)
            correct += (y_hat == y).sum().item()
            total += y.size(0)
    net.train()
    return correct / total

# Fonction pour entraîner le modèle
def train(net, optimizer, trainloader, testloader, epochs=10):
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        net.train()
        running_loss = []
        t = tqdm(trainloader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
            t.set_description(f'Epoch {epoch+1}, training loss: {sum(running_loss)/len(running_loss):.4f}')

        epoch_loss = sum(running_loss) / len(running_loss)

        # Test à la fin de chaque époque
        acc = test(net, testloader)

        print(f"Epoch {epoch+1} | loss: {epoch_loss:.4f} | test acc: {acc:.2%}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    # Transforms pour les posters
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = torchvision.datasets.ImageFolder(
        '../sorted_movie_posters_paligema',
        transform=transform
    )

    # Split train/test (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Instanciation du modèle
    net = MovieposterNet()
    net = net.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)

    # Training
    train(net, optimizer, trainloader, testloader, epochs=args.epochs)

    # Sauvegarde des poids
    torch.save(net.state_dict(), '../weights/movieposter_net.pth', _use_new_zipfile_serialization=False)