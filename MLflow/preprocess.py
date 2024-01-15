import pandas as pd
from numpy import ndarray


# выбирает из датафрейма все данные и возвращает массив типа numpy.ndarray
def preprocess_input(input_df: pd.DataFrame) -> ndarray:
    return input_df.values


# Возвращает данные (массив 0 1 2) в виде таблицы с колонкой ирисы
def preprocess_output(output_data: ndarray) -> pd.DataFrame:
    return pd.DataFrame(data=output_data, columns=['Irises'])
