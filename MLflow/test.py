import pickle

from mlflow.types.schema import Schema, ColSpec
import mlflow.pyfunc
import mlflow.transformers

import pandas as pd

from sklearn.datasets import load_iris


def create_iris_df() -> pd.DataFrame:
    iris = load_iris()
    return pd.DataFrame(data=iris.data, columns=iris.feature_names)


class Model(mlflow.pyfunc.PythonModel):
    def predict(self, context, iris_df):
        from preprocess import preprocess_input, preprocess_output
        input_data = preprocess_input(iris_df)

        with open(context.artifacts["sklearn_model"], 'rb') as f:
            cls = pickle.load(f)
        predicted_data = cls.predict(input_data)
        predicted_df = preprocess_output(predicted_data)

        return predicted_df


class Create_Model:
    def __init__(self):
        model_path = "model.pkl"
        artifacts = {
            "sklearn_model": model_path
        }

        input_schema = mlflow.types.Schema([
            ColSpec(type=mlflow.types.DataType.double, name='sepal length (cm)'),
            ColSpec(type=mlflow.types.DataType.double, name='sepal width (cm)'),
            ColSpec(type=mlflow.types.DataType.double, name='petal length (cm)'),
            ColSpec(type=mlflow.types.DataType.double, name='petal width (cm)'),
        ])

        output_schema = mlflow.types.Schema([
            ColSpec(type=mlflow.types.DataType.integer, name="Irises")
        ])

        model_signature = mlflow.models.signature.ModelSignature(
            inputs=input_schema,
            outputs=output_schema,
        )

        mlflow.pyfunc.save_model(
            path="Saved_model",
            python_model=Model(),
            artifacts=artifacts,
            signature=model_signature,
            code_path=["./preprocess.py"],
        )

if __name__ == "__main__":
    Create_Model()
