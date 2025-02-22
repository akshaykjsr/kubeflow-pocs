import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # Parse input arguments
    parser = argparse.ArgumentParser(description="Train an ML model with Kubeflow parameters")
    parser.add_argument("--model_type", type=str, required=True, help="Type of model (logistic_regression)")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")

    args = parser.parse_args()

    print(f"Training {args.model_type} with learning_rate={args.learning_rate}, batch_size={args.batch_size}")

    # Generate synthetic data
    X = np.random.rand(1000, 10)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train logistic regression model
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
