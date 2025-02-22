# **Kubeflow Pipeline for ML Model Training with Parameter Passing**

This repository demonstrates how to use **Kubeflow Pipelines (KFP)** to train an ML model while dynamically passing parameters from **Kubeflow UI** or **Python SDK**. The pipeline trains a **logistic regression model** and supports **hyperparameter tuning** via Kubeflow.

---

## **🚀 1. Prerequisites**
Before running the pipeline, ensure your system has the following installed:
Note: 
    1- this poc has been done on ubuntu os
    2- make sure docker have user added or else would need to sudo for every command even "docker ps"
    3- kubeflow versions are going to get mismatched if tried on virtual python env, so avoid for this poc
    4- make sure the docker login is already done and after doing "minikube ssh" also docker pull for the image is going to happen
    5- give it at least 20 mins during kubeflow installation for pods to start running and then then only move ahead

### **🔹 System Dependencies**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip docker.io docker-compose \
                    curl git jq unzip build-essential
```

### **🔹 Install & Configure Minikube (Local Kubernetes Cluster)**
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
minikube start --driver=docker
```

### **🔹 Install Kubeflow Pipelines (KFP) CLI**
```bash
pip3 install kfp
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/minikube?ref=1.8.5"
kubectl get pods -n kubeflow --watch
```
Wait until all pods are in `Running` state.

### **🔹 Start Kubeflow UI**
```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```
Now visit **[http://localhost:8080](http://localhost:8080)**.

---

## **🟢 2. Pipeline Components**
### **📌 2.1 `train_model.py` (ML Training Script)**
```python
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser(description="Train an ML model with Kubeflow parameters")
    parser.add_argument("--model_type", type=str, required=True, help="Type of model (logistic_regression)")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    args = parser.parse_args()
    
    print(f"Training {args.model_type} with learning_rate={args.learning_rate}, batch_size={args.batch_size}")
    X = np.random.rand(1000, 10)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    if args.model_type == "logistic_regression":
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
    else:
        print("Invalid model type!")

if __name__ == "__main__":
    main()
```

---

### **📌 2.2 `Dockerfile` (Custom Image for Training Script)**
```dockerfile
FROM python:3.8
WORKDIR /app
COPY train_model.py /app/train_model.py
RUN pip install numpy scikit-learn
ENTRYPOINT ["python3", "/app/train_model.py"]
```

### **Build & Push the Docker Image**
```bash
docker build -t your-dockerhub-user/ml-trainer:v1 .
docker login
docker push your-dockerhub-user/ml-trainer:v1
```

---

### **📌 2.3 `component.yaml` (Kubeflow Component Definition)**
```yaml
name: "Train ML Model"
description: "Trains an ML model with parameters from Kubeflow"
inputs:
  - {name: model_type, type: String, description: "Type of model (logistic_regression)"}
  - {name: learning_rate, type: Float, description: "Learning rate"}
  - {name: batch_size, type: Integer, description: "Batch size"}
implementation:
  container:
    image: your-dockerhub-user/ml-trainer:v1
    command: ["python3", "/app/train_model.py"]
    args: [
      "--model_type", {inputValue: model_type},
      "--learning_rate", {inputValue: learning_rate},
      "--batch_size", {inputValue: batch_size}
    ]
```

---

### **📌 2.4 `pipeline.py` (Kubeflow Pipeline Definition)**
```python
import kfp
from kfp import dsl
from kfp.components import load_component_from_file
from kfp.compiler import Compiler

train_model_op = load_component_from_file("component.yaml")

@dsl.pipeline(
    name="ML Training Pipeline",
    description="Trains an ML model with Kubeflow parameters"
)
def ml_training_pipeline(
    model_type: str = "logistic_regression",
    learning_rate: float = 0.01,
    batch_size: int = 32
):
    train_model_op(
        model_type=model_type,
        learning_rate=learning_rate,
        batch_size=batch_size
    )

if __name__ == "__main__":
    Compiler().compile(ml_training_pipeline, "ml_training_pipeline.yaml")
```

Compile the pipeline:
```bash
python3 pipeline.py
```

---

## **🟢 3. Upload & Run the Pipeline in Kubeflow**

### **1️⃣ Upload `ml_training_pipeline.yaml`**
1. Open **Kubeflow UI (`http://localhost:8080`)**.
2. Go to **Pipelines → Upload Pipeline**.
3. Upload `ml_training_pipeline.yaml` and click **Create**.

### **2️⃣ Run the Pipeline with Different Parameters**
1. Click **Create Run**.
2. Modify Parameters:
   - `learning_rate = 0.05`
   - `batch_size = 64`
3. Click **Start**.

---

## **🟢 4. Automate Pipeline Execution Using Kubeflow SDK**
Create `run_experiment.py`:
```python
import kfp
from kfp.client import Client

client = Client(host='http://localhost:8080')

client.create_run_from_pipeline_package(
    pipeline_file="ml_training_pipeline.yaml",
    arguments={
        "model_type": "logistic_regression",
        "learning_rate": 0.1,
        "batch_size": 128
    },
    experiment_name="ML Experiments",
    run_name="Run with modified params"
)
```
Run:
```bash
python3 run_experiment.py
```

---

## **🚀 Final Takeaways**
✅ **Kubeflow Pipelines enable parameterized ML training workflows.**  
✅ **Easily modify hyperparameters via UI or SDK for experimentation.**  
✅ **Reproducibility, automation, and tracking of ML experiments.**  

🚀 **This setup allows scalable, automated, and reproducible ML workflows in Kubernetes!** 🎯


