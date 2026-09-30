import os
import random 
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT = './data'


mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


full_train_data = datasets.CIFAR10(root=ROOT, train=True, download=True, transform=train_transforms)
test_data = datasets.CIFAR10(root=ROOT, train=False, download=True, transform=test_transforms)
valid_dataset = datasets.CIFAR10(root=ROOT, train=True, download=True, transform=test_transforms)
#Split
VALID_RATIO = 0.9
n_train_examples = int(len(full_train_data) * VALID_RATIO)
n_valid_examples = len(full_train_data) - n_train_examples

train_data, valid_data = data.random_split(
    full_train_data,
    [n_train_examples, n_valid_examples]
)

BATCH_SIZE = 128

train_dataloader = data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
valid_dataloader = data.DataLoader(valid_data, batch_size=BATCH_SIZE)
test_dataloader = data.DataLoader(test_data, batch_size=BATCH_SIZE)

#LeNet
class LeNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(16 * 6 * 6, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, inputs):
        outputs = F.relu(self.avgpool1(self.conv1(inputs)))
        outputs = F.relu(self.avgpool2(self.conv2(outputs)))
        outputs = self.flatten(outputs)
        outputs = F.relu(self.fc_1(outputs))
        outputs = F.relu(self.fc_2(outputs))
        outputs = self.fc_3(outputs)
        return outputs

#Train
def train(model, optimizer, criterion, train_dataloader, device, epoch=0, log_interval=50):
    model.train()
    epoch_acc, epoch_count = 0, 0
    running_acc, running_count = 0, 0
    losses = []
    start_time = time.time()

    for idx, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, labels)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        correct = (predictions.argmax(1) == labels).sum().item()
        batch_size = labels.size(0)

        epoch_acc += correct
        epoch_count += batch_size
        running_acc += correct
        running_count += batch_size

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(f"| epoch {epoch:3d} | {idx:5d}/{len(train_dataloader):5d} batches | accuracy {running_acc / running_count:8.3f}")
            running_acc, running_count = 0, 0
            start_time = time.time()

    return epoch_acc / epoch_count, sum(losses) / len(losses)

# Evaluate
def evaluate(model, criterion, dataloader, device):
    model.eval()
    total_acc, total_count = 0, 0
    losses = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            predictions = model(inputs)
            loss = criterion(predictions, labels)
            losses.append(loss.item())

            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len (losses)
    return epoch_acc, epoch_loss




num_classes = len(train_data.dataset.classes)

lenet_model = LeNetClassifier(num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(lenet_model.parameters())

num_epochs = 10
save_model = './model'
os.makedirs(save_model, exist_ok=True)

model_path = os.path.join(save_model, 'lenet_model_CIFAR10.pt')

train_accs, train_losses = [], []
eval_accs, eval_losses = [], []
best_loss_eval = float('inf')

for epoch in range(1, num_epochs + 1):
    epoch_start_time = time.time()
    
    train_acc, train_loss = train(lenet_model, optimizer, criterion, train_dataloader, device, epoch)
    train_accs.append(train_acc)
    train_losses.append(train_loss)

    eval_acc, eval_loss = evaluate(lenet_model, criterion, valid_dataloader, device)
    eval_accs.append(eval_acc)
    eval_losses.append(eval_loss)

   
    if eval_loss < best_loss_eval:
        best_loss_eval = eval_loss
        torch.save(lenet_model.state_dict(), model_path)
        print(f"--> Đã lưu checkpoint tốt nhất tại epoch {epoch} (Val Loss: {eval_loss:.4f})")

    print(f"Epoch {epoch:2d} | Train Acc: {train_acc:.4f} | Val Acc: {eval_acc:.4f} | Thời gian: {time.time() - epoch_start_time:.2f}s")
    print("-" * 59)

test_data.transform = test_transforms
test_dataloader = data.DataLoader(
    test_data,
    batch_size= BATCH_SIZE
)
test_acc, test_loss = evaluate(lenet_model, criterion, test_dataloader, device)
test_acc, test_loss