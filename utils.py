import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F
import cv2
import random
import numpy as np
from math import floor
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,   64,  3)
        self.conv2 = nn.Conv2d(64,  128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(128, 256)
        self.drop3 = nn.Dropout2d(p=0.5)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.drop3(x)
        x = self.fc3(x)
        return x


def load_data():
    """Load CIFAR-10 (training and test set)."""
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])

    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform_train)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform_test)
    trainset.targets, testset.targets = torch.LongTensor(trainset.targets), torch.LongTensor(testset.targets)

    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainset, testset, num_examples


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)
    trainset, testset, num_examples = load_data()
    return (trainset, testset)
    #n_train = int(num_examples["trainset"] / 10)
    #n_test = int(num_examples["testset"] / 10)

    #train_parition = torch.utils.data.Subset(
    #    trainset, range(idx * n_train, (idx + 1) * n_train)
    #)
    #test_parition = torch.utils.data.Subset(
    #    testset, range(idx * n_test, (idx + 1) * n_test)
    #)
    #return (train_parition, test_parition)

class DatasetSplit(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])

    def classes(self):
        return torch.unique(self.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        inp, target = self.dataset[self.idxs[item]]
        return inp, target


def poison_dataset(dataset, data_idxs=None, poison_all=False, agent_idx=-1):
    all_idxs = (dataset.targets == 5).nonzero().flatten().tolist()
    if data_idxs != None:
        all_idxs = list(set(all_idxs).intersection(data_idxs))

    #poison_frac = 1 if poison_all else 0.5
    poison_frac = 0.5
    #print("Poinson fraction: " + str(poison_frac))
    poison_idxs = random.sample(all_idxs, floor(poison_frac*len(all_idxs)))
    #print("Poisoning {} images".format(len(poison_idxs)))
    for idx in poison_idxs:
        #if args.data == 'fedemnist':
        #    clean_img = dataset.inputs[idx]
        #else:
        clean_img = dataset.data[idx]
        #print("pre: " + str(clean_img.shape))
        #test_image = clean_img.transpose(2,1,0)
        #print("post: " + str(test_image.shape))
        #plt.imshow(clean_img)
        #plt.title("test")
        #print(clean_img)
        bd_img = add_pattern_bd(clean_img, 'cifar10', pattern_type='plus', agent_idx=agent_idx)
        #if args.data == 'fedemnist':
        #     dataset.inputs[idx] = torch.tensor(bd_img)
        #else:
        dataset.data[idx] = torch.tensor(bd_img)
        
        dataset.targets[idx] = 7
    return


def add_pattern_bd(x, dataset='cifar10', pattern_type='square', agent_idx=-1):
    """
    adds a trojan pattern to the image
    """
    x = np.array(x.squeeze())

    # if cifar is selected, we're doing a distributed backdoor attack (i.e., portions of trojan pattern is split between agents, only works for plus)
    if dataset == 'cifar10':
        if pattern_type == 'plus':
            start_idx = 5
            size = 6
            if agent_idx == -1:
                # vertical line
                for d in range(0, 3):
                    for i in range(start_idx, start_idx+size+1):
                        #print("changing {} to 0".format(str(x[i, start_idx][d])))
                        x[i, start_idx][d] = 0
                # horizontal line
                for d in range(0, 3):
                    for i in range(start_idx-size//2, start_idx+size//2 + 1):
                        x[start_idx+size//2, i][d] = 0
            else:# DBA attack
                #upper part of vertical
                if agent_idx % 4 == 0:
                    for d in range(0, 3):
                        for i in range(start_idx, start_idx+(size//2)+1):
                            x[i, start_idx][d] = 0

                #lower part of vertical
                elif agent_idx % 4 == 1:
                    for d in range(0, 3):
                        for i in range(start_idx+(size//2)+1, start_idx+size+1):
                            x[i, start_idx][d] = 0

                #left-part of horizontal
                elif agent_idx % 4 == 2:
                    for d in range(0, 3):
                        for i in range(start_idx-size//2, start_idx+size//4 + 1):
                            x[start_idx+size//2, i][d] = 0

                #right-part of horizontal
                elif agent_idx % 4 == 3:
                    for d in range(0, 3):
                        for i in range(start_idx-size//4+1, start_idx+size//2 + 1):
                            x[start_idx+size//2, i][d] = 0

    elif dataset == 'fmnist':
        if pattern_type == 'square':
            for i in range(21, 26):
                for j in range(21, 26):
                    x[i, j] = 255

        elif pattern_type == 'copyright':
            trojan = cv2.imread('../watermark.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            x = x + trojan

        elif pattern_type == 'apple':
            trojan = cv2.imread('../apple.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            x = x + trojan

        elif pattern_type == 'plus':
            start_idx = 5
            size = 5
            # vertical line
            for i in range(start_idx, start_idx+size):
                x[i, start_idx] = 255

            # horizontal line
            for i in range(start_idx-size//2, start_idx+size//2 + 1):
                x[start_idx+size//2, i] = 255

    elif dataset == 'fedemnist':
        if pattern_type == 'square':
            for i in range(21, 26):
                for j in range(21, 26):
                    x[i, j] = 0

        elif pattern_type == 'copyright':
            trojan = cv2.imread('../watermark.png', cv2.IMREAD_GRAYSCALE)

        elif pattern_type == 'apple':
            trojan = cv2.imread('../apple.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)/255
            x = x - trojan

        elif pattern_type == 'plus':
            start_idx = 8
            size = 5
            # vertical line
            for i in range(start_idx, start_idx+size):
                x[i, start_idx] = 0

            # horizontal line
            for i in range(start_idx-size//2, start_idx+size//2 + 1):
                x[start_idx+size//2, i] = 0

    return x   



def train(net, trainloader, valloader, poinsonedloader, epochs, device: str = "cpu"):
    """Train the network on the training set."""
    print("Starting training...")
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.1, momentum=0.9 #, weight_decay=1e-4
    )
    scalar = torch.cuda.amp.GradScaler()
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            #print("\nnet(images): " + str(net(images).shape))
            #print("labels: " + str(labels.shape) + "\n")
            #with autocast():
            loss = criterion(net(images), labels)
            scalar.scale(loss).backward()
            #loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10)
            scalar.step(optimizer)
            
            scalar.update()
            #optimizer.step()

    net.to("cpu")  # move model back to CPU

    print("train eval")
    train_loss, train_acc, train_per_class = test(net, trainloader, None, device)
    print("val eval")
    val_loss, val_acc, val_per_class = test(net, valloader, None, device)
    #print("poison eval")
    #poison_loss, poison_acc, poison_per_class = test(net, poinsonedloader, None, device)
    #val_loss, val_acc = test(net, trainloader)

    #print("Length of trainset: " + str(len(trainloader.dataset)))
    #print("Length of validation set: " + str(len(valloader.dataset)))
    #print("Length of poison set: " + str(len(poinsonedloader.dataset)))

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        #"train_accuracy_per_class": train_per_class,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        #"val_accuracy_per_class": val_per_class,
        #"poison_loss": poison_loss,
        #"poison_accuracy": poison_acc,
        #"poison_accuracy_per_class": poison_per_class,
    }
    return results


def test(net, testloader, steps: int = None, device: str = "cpu"):
    """Validate the network on the entire test set."""
    print("Starting evalutation...")
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    loss, accuracy, per_class_accuracy = get_loss_and_accuracy(net, criterion, testloader, steps, device)
    net.to("cpu")  # move model back to CPU
    return loss, accuracy, per_class_accuracy

def get_loss_and_accuracy(model, criterion, data_loader, steps: int = None, device: str = "cpu"):
    correct, loss = 0, 0.0
    model.eval()
    confusion_matrix = torch.zeros(10, 10)
    #print("\ttest1")
    with torch.no_grad():
        #print("\ttest2")
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            #print("Outputs")
            #print(outputs.data)
            #print("Labels")
            #print(labels)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.view(-1)
            correct += (predicted == labels).sum().item()
            if steps is not None and batch_idx == steps:
                break
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    accuracy = correct / len(data_loader.dataset)
    avg_loss = loss / len(data_loader.dataset)
    print("\tAvg Loss: {:.3f}".format(avg_loss))
    print("\tAccuracy: " + str(accuracy))
    print("\tPer class accuracy: " + str(per_class_accuracy))
    return avg_loss, accuracy, per_class_accuracy


def replace_classifying_layer(efficientnet_model, num_classes: int = 10):
    """Replaces the final layer of the classifier."""
    num_features = efficientnet_model.classifier.fc.in_features
    efficientnet_model.classifier.fc = torch.nn.Linear(num_features, num_classes)



def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]