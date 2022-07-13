
import pandas as pd

def csv_to_json(csv_path):
    csv_data = pd.read_csv(csv_path, sep = ",")
    csv_data.to_json("test.json")

def json_to_csv(json_path):

    json_str = '{"A":{"0":"B","1":"D"},"Unnamed: 1":{"0":null,"1":"E"},"Unnamed: 2":{"0":"C","1":"F"}}'

    # json_str = '[{"A": "B", "Unnamed: 1": null, "Unnamed: 2": "C"}, {"A": "D", "Unnamed: 1": "E", "Unnamed: 2": "F"}]'
    df = pd.read_json(json_str)
    print(df)

# csv_to_json("/home/jinho/Downloads/tmp.csv")
json_to_csv('asd')
