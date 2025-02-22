import kfp
from kfp import dsl
from kfp.components import load_component_from_file
from kfp.compiler import Compiler

# Load the component
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
