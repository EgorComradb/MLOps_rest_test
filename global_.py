import pathlib
class paths:
    path_to_save_model = str(pathlib.Path(__file__).resolve().parent) + "\MLflow\Saved_model"
    path_to_model = str(pathlib.Path(__file__).resolve().parent) + "\MLflow\model.pkl"


