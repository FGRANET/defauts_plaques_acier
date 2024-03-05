import pandas as pd

def load_csv_data(file_path):
    """
    Charge les données depuis un fichier CSV.
    
    :param file_path: str, le chemin vers le fichier CSV.
    :return: pandas.DataFrame, les données chargées.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
        return None