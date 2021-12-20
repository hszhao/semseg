import os

import fire
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

import acosp.inject
import util.util
from acosp.pruner import SoftTopKPruner


def train(model, gpu, optimizer, criterion, train_loader, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images, labels = images.to(gpu), labels.to(gpu)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f"Epoch {epoch}: Avg train loss: {train_loss / total:.3} | Acc: {100. * correct / total:.4}")


def test(model, gpu, criterion, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for id, (images, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images, labels = images.to(gpu), labels.to(gpu)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = test_loss / total
    avg_acc = 100.0 * correct / total
    print(f"Epoch {epoch}: Avg test Loss: {avg_loss:.3} | Acc: {avg_acc:.4}")

    return avg_loss, avg_acc


def main(
    save_dir,
    gpu: int = 0,
    lr: float = 0.01,
    batch_size=512,
    epochs=50,
    sparsity: float = 0,
    pruner_ending_epoch=10,
):
    model = models.resnet18(num_classes=10, pretrained=False)
    # modify the neck of the ResNet module and replace the Convolution layer, because 7x7 convolution with stride=2
    # is not a good start for small images.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1), bias=False)

    pruner = SoftTopKPruner(
        starting_epoch=0,
        ending_epoch=pruner_ending_epoch,
        final_sparsity=sparsity,
        active=sparsity > 0,
    )
    pruner.configure_model(model)

    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)

    train_dataset = datasets.CIFAR10(
        "cifar10",
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]
        ),
        target_transform=lambda x: torch.tensor(x),
    )

    test_dataset = datasets.CIFAR10(
        "cifar10",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]
        ),
        target_transform=lambda x: torch.tensor(x),
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    os.makedirs(save_dir, exist_ok=True)
    results_fp = os.path.join(save_dir, f"log_{sparsity}.txt")
    with open(results_fp, "w+") as file:
        file.write("epoch, loss, acc\n")

    best_acc = 0
    for epoch in range(epochs):
        if epoch == pruner.ending_epoch:
            acosp.inject.soft_to_hard_k(model)

        optimizer.param_groups[0]["lr"] = util.util.cosine_learning_rate(lr, epoch, epochs)

        train(model, gpu, optimizer, criterion, train_loader, epoch)

        loss, acc = test(model, gpu, criterion, test_loader, epoch)
        with open(results_fp, "a") as file:
            file.write(f"{epoch}, {loss}, {acc}\n")

        pruner.update_mask_layers(model, epoch)

        if acc > best_acc:
            torch.save(model, os.path.join(save_dir, f"model_{sparsity}.pth"))
            best_acc = acc


if __name__ == "__main__":
    fire.Fire(main)
