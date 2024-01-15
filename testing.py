from pprint import pprint
from io import StringIO

import requests
from sklearn.datasets import load_iris
import pandas as pd

def post_req():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    js_frame = df.to_json()

    test = requests.post(url="http://127.0.0.1:8000/set_flower_params", json=js_frame)
    print(pd.read_json(StringIO(test.json())))

def get_in_data():
    test = requests.get(url="http://127.0.0.1:8000/get_input_column")
    pprint(test.json())

def get_out_data():
    test = requests.get(url="http://127.0.0.1:8000/get_output_column")
    pprint(test.json())

post_req()
get_in_data()
get_out_data()

