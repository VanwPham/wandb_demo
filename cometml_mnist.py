import comet_ml
from comet_ml import Optimizer

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve

WORKSPACE    = "vanwpham"        
PROJECT_NAME = "mnist-comet-demo"

sweep_parameters = {
    "learning_rate": {"type": "float", "min": 1e-5, "max": 1e-1},
    "momentum":      {"type": "float", "min": 0.5,  "max": 0.99},
    "batch_size":    {"type": "categorical", "values": ["32", "64", "128"]},
}

sweep_config = {
    "algorithm": "bayes",
    "parameters": sweep_parameters,
    "spec": {
        "metric": {"name": "val_accuracy", "goal": "maximize"},
        "trials": 10
    }
}

optimizer = Optimizer(
    api_key=os.getenv("COMET_API_KEY"),
    workspace=WORKSPACE,
    project_name=PROJECT_NAME,
    config=sweep_config
)

def train_fn(experiment, params):
    experiment.log_parameters(params)
    lr       = params["learning_rate"]
    momentum = params["momentum"]
    bs       = int(params["batch_size"])
    epochs   = 5

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST("data", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST("data", train=False, download=True, transform=transform)

    train_counts = pd.Series(train_ds.targets.numpy()).value_counts().sort_index()
    test_counts  = pd.Series(test_ds.targets.numpy()).value_counts().sort_index()
    df_sum = pd.DataFrame({
        "label":       range(10),
        "train_count": train_counts.values,
        "test_count":  test_counts.values
    })
    # FIX lỗi log_table và step bắt buộc của histogram
    experiment.log_table("dataset_summary.csv", df_sum)
    experiment.log_histogram_3d(train_counts.values, name="train_dist", step=0)
    experiment.log_histogram_3d(test_counts.values,  name="test_dist", step=0)

    sample_imgs = torch.stack([train_ds[i][0] for i in range(16)])
    grid = make_grid(sample_imgs, nrow=4)
    grid_np = grid.permute(1,2,0).numpy()
    experiment.log_image(grid_np, name="sample_images")

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1,16,3,1)
            self.conv2 = nn.Conv2d(16,32,3,1)
            self.fc1   = nn.Linear(32*5*5,128)
            self.fc2   = nn.Linear(128,10)
        def forward(self, x):
            x = nn.functional.relu(self.conv1(x))
            x = nn.functional.max_pool2d(x,2)
            x = nn.functional.relu(self.conv2(x))
            x = nn.functional.max_pool2d(x,2)
            x = x.flatten(1)
            x = nn.functional.relu(self.fc1(x))
            return self.fc2(x)

    model     = SimpleCNN()
    optimizer_model = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    experiment.watch(model, log_graph=True, log_weights=True, log_gradients=True)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False)

    best_val_acc = 0.0
    ckpt = None

    for epoch in range(1, epochs+1):
        model.train()
        loss_sum, correct = 0.0, 0
        start = time.time()
        for X, y in train_loader:
            optimizer_model.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer_model.step()
            loss_sum += loss.item()
            correct  += (out.argmax(1)==y).sum().item()
        train_acc  = correct / len(train_loader.dataset)
        train_time = time.time() - start

        model.eval()
        val_loss, val_correct = 0.0, 0
        preds, trues = [], []
        with torch.no_grad():
            for X, y in test_loader:
                out = model(X)
                val_loss   += criterion(out, y).item()
                p = out.argmax(1)
                val_correct += (p==y).sum().item()
                preds.extend(p.cpu().numpy())
                trues.extend(y.cpu().numpy())
        val_acc   = val_correct / len(test_loader.dataset)
        precision = precision_score(trues, preds, average='macro')
        recall    = recall_score(trues, preds, average='macro')
        f1        = f1_score(trues, preds, average='macro')

        experiment.log_confusion_matrix(y_true=trues, y_pred=preds, step=epoch)
        try:
            fpr, tpr, _ = roc_curve([t==0 for t in trues], [p==0 for p in preds])
            experiment.log_curve(fpr, tpr, name="roc_curve_class0", step=epoch)
        except Exception:
            pass

        experiment.log_metrics({
            "train_loss": loss_sum/len(train_loader),
            "train_acc":  train_acc,
            "val_loss":   val_loss/len(test_loader),
            "val_acc":    val_acc,
            "precision":  precision,
            "recall":     recall,
            "f1_score":   f1,
            "train_time": train_time
        }, epoch=epoch, step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = f"best_{experiment.get_key()}_e{epoch}.pth"
            torch.save(model.state_dict(), ckpt)
            experiment.log_model("best_checkpoint", ckpt)

    if ckpt:
        experiment.log_model("final_model", ckpt)
    report_md = f"# MNIST CometML Demo Report\n\nBest Val Accuracy: {best_val_acc:.4f}"
    experiment.log_text(report_md, file_name="report.md")
    experiment.end()

if __name__ == "__main__":
    for experiment in optimizer.get_experiments():
        params = experiment.params
        train_fn(experiment, params)
