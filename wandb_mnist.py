import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb
import yaml

# Load sweep config từ file YAML
with open("./sweep_config.yaml") as f:
    print("Loading sweep config...")
    sweep_config = yaml.safe_load(f)
    print("Sweep config loaded:", sweep_config)
    # Ép kiểu float cho các giá trị min, max
    sweep_config['parameters']['learning_rate']['min'] = float(sweep_config['parameters']['learning_rate']['min'])
    sweep_config['parameters']['learning_rate']['max'] = float(sweep_config['parameters']['learning_rate']['max'])
    sweep_config['parameters']['momentum']['min'] = float(sweep_config['parameters']['momentum']['min'])
    sweep_config['parameters']['momentum']['max'] = float(sweep_config['parameters']['momentum']['max'])

def train():
    wandb.init(project="mnist-demo-final", tags=["sweep", "media-demo", "model-registry"])
    config = wandb.config

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Log bảng phân phối nhãn dữ liệu
    train_counts = pd.Series(train_ds.targets.numpy()).value_counts().sort_index()
    test_counts = pd.Series(test_ds.targets.numpy()).value_counts().sort_index()
    df_summary = pd.DataFrame({
        "label": list(range(10)),
        "train_count": train_counts.values,
        "test_count": test_counts.values
    })
    
    imgs = torch.stack([train_ds[i][0] for i in range(16)])
    mean = 0.1307
    std = 0.3081
    imgs = imgs * std + mean
    imgs_uint8 = (imgs * 255).clamp(0, 255).to(torch.uint8)
    img_grid = make_grid(imgs_uint8, nrow=4)
    
    wandb.log({
        "dataset_table": wandb.Table(dataframe=df_summary),
        "bar_train": wandb.plot.bar(wandb.Table(dataframe=df_summary), "label", "train_count", title="Train label distribution"),
        "bar_test": wandb.plot.bar(wandb.Table(dataframe=df_summary), "label", "test_count", title="Test label distribution"),
        "image_grid": wandb.Image(img_grid, caption="Sample images"),
    })

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, 1)
            self.conv2 = nn.Conv2d(16, 32, 3, 1)
            self.fc1 = nn.Linear(32 * 5 * 5, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = nn.functional.relu(self.conv1(x))
            x = nn.functional.max_pool2d(x, 2)
            x = nn.functional.relu(self.conv2(x))
            x = nn.functional.max_pool2d(x, 2)
            x = x.view(-1, 32 * 5 * 5)
            x = nn.functional.relu(self.fc1(x))
            return self.fc2(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    criterion = nn.CrossEntropyLoss()
    wandb.watch(model, log="all", log_freq=50)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    best_val_acc = 0.0
    best_ckpt = None

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        preds = []
        trues = []
        images_to_log = []
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                preds.extend(pred.cpu().numpy())
                trues.extend(target.cpu().numpy())

                if idx == 0:
                    imgs = data[:16].cpu()
                    gt_labels = target[:16].cpu().numpy()
                    pred_labels = pred[:16].cpu().numpy()

                    for i in range(16):
                        img_uint8 = (imgs[i] * 255).clamp(0, 255).to(torch.uint8)
                        caption = f"GT: {gt_labels[i]}, Pred: {pred_labels[i]}"
                        images_to_log.append(wandb.Image(img_uint8, caption=caption))

        val_loss /= len(test_loader.dataset)
        val_acc = correct / len(test_loader.dataset)

        precision = precision_score(trues, preds, average='macro', zero_division=0)
        recall = recall_score(trues, preds, average='macro', zero_division=0)
        f1 = f1_score(trues, preds, average='macro', zero_division=0)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "val_images": images_to_log
        })

        # Lưu artifact mô hình nếu tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = f"mnist_best_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            artifact = wandb.Artifact(name="mnist-model-best", type="model",
                                     metadata={"epoch": epoch + 1, "val_acc": val_acc})
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)
            os.remove(ckpt_path)  # Xoá file sau khi upload

    wandb.finish()

def register_model():
    """Chạy riêng sau khi hoàn thành các run, để đăng ký model artifact"""
    api = wandb.Api()
    entity = "cappy"  # đổi thành entity của bạn
    project = "mnist-demo-final"
    artifact_name = "mnist-model-best:latest"
    artifact = api.artifact(f"{entity}/{project}/{artifact_name}", type="model")
    model_registry = artifact.register()
    model_registry.transition("staging")  # hoặc "production"
    print(f"Artifact {artifact_name} đã được đăng ký và chuyển trạng thái staging")

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="mnist-demo-final")
    wandb.agent(sweep_id, function=train, count=7)

    # Khi chạy xong các run, bạn gọi riêng hàm register_model() hoặc tạo script riêng để chạy
    # register_model()
