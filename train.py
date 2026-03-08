import argparse
from statistics import mean

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

from model import MovieposterNet

 # setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(net, optimizer, loader, writer,epochs=10):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}')
        writer.add_scalar('training loss', mean(running_loss), epoch)


def test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total
	
if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default = 'Movieposter', help='experiment name')
    parser.add_argument('--epochs', type=int, default = int(10), help='nb of epochs')
    parser.add_argument('--batch_size', type=int, default = int(64), help='batch size')
    parser.add_argument('--lr', type=float, default =  float(1e-3), help='learning rate')


    args = parser.parse_args()
    print(args.exp_name)
    exp_name = args.exp_name
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    writer = SummaryWriter(f'runs/Movieposter')

    # 1. Définition des transformations
    # Les posters sont en couleur (3 canaux) et de tailles variées, contrairement à MNIST.
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Redimensionnement standard pour les modèles de vision
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalisation sur 3 canaux (RGB)
    ])

    # 2. Chargement du dataset complet
    # Le chemin '../' permet de remonter d'un niveau par rapport au dossier 'projet_AIF'
    data_dir = '../sorted_movie_posters_paligema'
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # 3. Division en train/test (ex: 80% train, 20% test)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    trainset, testset = random_split(full_dataset, [train_size, test_size])

    # 4. Création des DataLoaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Accès aux classes (genres)
    classes = full_dataset.classes
    print(f"Classes détectées : {classes}")


    net =MovieposterNet().to(device)

    # setting net on device(GPU if available, else CPU)
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)

    train(net, optimizer,trainloader, writer, epochs)
    test_acc = test(net,testloader)
    print(f'Test accuracy: {test_acc}')   

    # 1. Gestion du dossier de sauvegarde des poids
    import os
    if not os.path.exists('weights'):
        os.makedirs('weights')
    
    torch.save(net.state_dict(), 'weights/movieposter_net.pth')

    # 2. Récupération d'un échantillon de données pour TensorBoard
    # On utilise le loader pour obtenir des tenseurs déjà transformés
    dataiter = iter(trainloader)
    images, labels = next(dataiter) 
    
    # On limite à 64 images pour la visualisation et on envoie sur le device
    images = images[:64].to(device)
    labels = labels[:64].to(device)

    # 3. Enregistrement du graphe du modèle
    # Vérifiez que les dimensions d'entrée du modèle correspondent (ex: 3, 224, 224)
    writer.add_graph(net, images)

    # 4. Enregistrement d'une grille d'images
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image('movieposter_samples', img_grid)

    # 5. Projecteur d'embeddings
    # get_features() doit être définie dans MovieposterNet pour retourner l'avant-dernière couche
    with torch.no_grad():
        try:
            embeddings = net.get_features(images)
            # Conversion des indices en noms de classes pour la lisibilité
            metadata = [classes[l] for l in labels]
            writer.add_embedding(embeddings,
                                metadata=metadata,
                                label_img=images, 
                                global_step=epochs)
        except AttributeError:
            print("Erreur : La méthode get_features n'est pas définie dans MovieposterNet.")

    # 6. Fermeture du SummaryWriter
    writer.close()