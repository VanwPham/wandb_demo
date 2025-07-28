import os
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
import pynvml
pynvml.nvmlInit()
import time
import numpy as np
import mlflow
import mlflow.pytorch
import mlflow.system_metrics as sm
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.tracking import MlflowClient
from mlflow.data import from_numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. Cấu hình MLflow
mlflow.set_tracking_uri("file:///home/van/Documents/mlruns")
mlflow.set_experiment("mnist-mlflow-demo-extended")

# 2. Cấu hình system metrics
sm.enable_system_metrics_logging()
sm.set_system_metrics_sampling_interval(5)       # Lấy mẫu mỗi 5s
sm.set_system_metrics_samples_before_logging(2)  # Log sau mỗi 2 mẫu

# 3. Đảm bảo không có run cũ dangling
if mlflow.active_run() is not None:
    mlflow.end_run()

# 4. Chuẩn bị dữ liệu và tạo objects để track datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds     = datasets.MNIST("data", train=True,  download=True, transform=transform)
test_ds      = datasets.MNIST("data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

# Chuyển sang numpy để log dataset lineage
X_train = train_ds.data.numpy().astype(np.float32) / 255.0
y_train = train_ds.targets.numpy()
X_test  = test_ds.data.numpy().astype(np.float32) / 255.0
y_test  = test_ds.targets.numpy()

train_input_ds = from_numpy(X_train, targets=y_train, name="mnist_train")
val_input_ds   = from_numpy(X_test,  targets=y_test,  name="mnist_validation")

# 5. Định nghĩa mô hình
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1   = nn.Linear(32*5*5, 128)
        self.fc2   = nn.Linear(128, 10)
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 32*5*5)
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)

# 6. Hàm train/validate với trace spans
def train_one_epoch(model, loader, optimizer, criterion):
    with mlflow.start_span(name="train_epoch", span_type="train") as span:
        span.set_inputs({"batch_size": loader.batch_size})
        start = time.time()
        model.train()
        total_loss, total_correct = 0.0, 0
        for X, y in loader:
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (out.argmax(1) == y).sum().item()
        duration = time.time() - start
        span.set_outputs({
            "epoch_loss":     total_loss / len(loader),
            "epoch_accuracy": total_correct / len(loader.dataset),
            "duration":       duration
        })
        return total_loss / len(loader), total_correct / len(loader.dataset), duration

def validate(model, loader, criterion):
    with mlflow.start_span(name="validate", span_type="eval") as span:
        model.eval()
        total_loss, total_correct = 0.0, 0
        preds_list, targets_list = [], []
        with torch.no_grad():
            for X, y in loader:
                out = model(X)
                total_loss += criterion(out, y).item()
                preds = out.argmax(1)
                total_correct += (preds == y).sum().item()
                preds_list.extend(preds.cpu().numpy())
                targets_list.extend(y.cpu().numpy())
        span.set_outputs({
            "val_loss":     total_loss / len(loader),
            "val_accuracy": total_correct / len(loader.dataset)
        })
        return total_loss / len(loader), total_correct / len(loader.dataset), preds_list, targets_list

def plot_confusion_matrix(y_true, y_pred, epoch):
    cm = pd.crosstab(pd.Series(y_true, name='Actual'),
                     pd.Series(y_pred, name='Predicted'))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix Epoch {epoch}')
    plt.tight_layout()
    fname = f"confusion_matrix_epoch{epoch}.png"
    plt.savefig(fname); plt.close()
    mlflow.log_artifact(fname, artifact_path="confusion_matrices")

# 7. Chạy một MLflow run duy nhất, bật system metrics cho run này
with mlflow.start_run(log_system_metrics=True) as run:
    # 7.1. Log dataset lineage
    mlflow.log_input(train_input_ds, context="training")
    mlflow.log_input(val_input_ds,   context="validation")

    # 7.2. Dataset summary & distribution
    tc = pd.Series(y_train).value_counts().sort_index()
    vc = pd.Series(y_test).value_counts().sort_index()
    ds_summary = pd.DataFrame({
        'class':       list(range(10)),
        'train_count': tc.values,
        'test_count':  vc.values
    })
    ds_summary.to_csv("dataset_summary.csv", index=False)
    mlflow.log_artifact("dataset_summary.csv", artifact_path="dataset_info")
    plt.figure(figsize=(8,4))
    plt.plot(ds_summary['class'], ds_summary['train_count'], marker='o', label='train')
    plt.plot(ds_summary['class'], ds_summary['test_count'],  marker='s', label='test')
    plt.xlabel('Class'); plt.ylabel('Count'); plt.title('MNIST Distribution'); plt.legend()
    plt.savefig("class_distribution.png"); plt.close()
    mlflow.log_artifact("class_distribution.png", artifact_path="dataset_info")

    # 7.3. Log params & tags
    mlflow.set_tag("model_type", "SimpleCNN")
    mlflow.log_params({"optimizer": "SGD", "momentum": 0.9})

    model     = SimpleCNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # 7.4. Training & validation loop
    for epoch in range(1, 6):
        tr_loss, tr_acc, tr_time = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, preds, trues = validate(model, test_loader, criterion)

        precision = precision_score(trues, preds, average='macro')
        recall    = recall_score(trues, preds, average='macro')
        f1        = f1_score(trues, preds, average='macro')

        mlflow.log_metrics({
            'train_loss':     tr_loss,
            'train_accuracy': tr_acc,
            'val_loss':       val_loss,
            'val_accuracy':   val_acc,
            'precision':      precision,
            'recall':         recall,
            'f1_score':       f1,
            'train_time':     tr_time
        }, step=epoch)

        plot_confusion_matrix(trues, preds, epoch)
        for name, param in model.named_parameters():
            mlflow.log_metric(f"weight_norm/{name}", param.norm().item(), step=epoch)

    # 7.5. Log & auto-register model
    example_batch, _ = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        example_out = model(example_batch)
    signature = infer_signature(example_batch.cpu().numpy(),
                                example_out.cpu().numpy())

    mlflow.pytorch.log_model(
        model,
        name="model",
        signature=signature,
        input_example=example_batch[:5].cpu().numpy(),
        registered_model_name="MNIST_SimpleCNN",
        await_registration_for=0
    )

    # 7.6. Log environment & sample predictions
    conda_env = _mlflow_conda_env(additional_pip_deps=[
        "torch", "torchvision", "matplotlib", "seaborn", "scikit-learn"
    ])
    mlflow.log_dict(conda_env, "conda_env.yaml")
    df_pred = pd.DataFrame({"actual": trues[:100], "predicted": preds[:100]})
    df_pred.to_csv("sample_predictions.csv", index=False)
    mlflow.log_artifact("sample_predictions.csv", artifact_path="predictions")

    print(f"Done. Run ID: {run.info.run_id}")
    