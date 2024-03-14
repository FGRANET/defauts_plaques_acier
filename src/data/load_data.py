import pandas as pd
from pathlib import Path

def load_csv_data(data_filename, data_folder_path):
    """
    Charge les données depuis un fichier CSV situé dans le dossier 'data' ou un de ses sous-dossiers.
    
    :param data_filename: str, le nom du fichier CSV à charger avec son extension .csv.
    :param data_folder: str, le chemin du dossier parent du projet.
    :return: pandas.DataFrame, les données chargées, ou None si le fichier n'est pas trouvé.
    """
    data_folder_path = Path(data_folder_path)
    # Rechercher le fichier dans le dossier et ses sous-dossiers
    file_list = list(data_folder_path.rglob(data_filename))
    
    # Vérifier si au moins un fichier correspondant au nom a été trouvé
    if file_list:
        # Charger les données depuis le premier fichier trouvé
        df = pd.read_csv(file_list[0], index_col=0)
        print(f"Data loaded successfully from {file_list[0]}")
        return df
    else:
        print(f"Le fichier {data_filename} n'a pas été trouvé dans {data_folder_path} ou ses sous-dossiers.")
        return None
