from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv_data(filename):
    """
    Charge les données depuis un fichier CSV situé dans le dossier 'data' relatif 
    à l'emplacement du script actuel.
    
    :param filename: str, le nom du fichier CSV à charger.
    :return: pandas.DataFrame, les données chargées, ou None si le fichier n'est pas trouvé.
    """
    # Obtenir le chemin du dossier où se trouve le script actuellement exécuté
    script_location = Path(__file__).resolve().parent
    
    # Construire un chemin relatif vers le fichier de données
    data_file_path = script_location / '..' /'..' / 'data' /'raw'/ filename
    
    # Assurer que le chemin est résolu (absolu) et que le fichier existe
    data_file_path = data_file_path.resolve()

    # Charger les données si le fichier existe
    if data_file_path.exists():
        try:
            data = pd.read_csv(data_file_path, index_col=0)
            print(f"Data loaded successfully from {data_file_path}")
            return data
        except Exception as e:
            print(f"Failed to load data from {data_file_path}: {e}")
    else:
        print(f"The file at {data_file_path} was not found.")
    
        return None


def clean_data(data):
    """Nettoie les données chargées."""
    # Exemple : Supprimer les lignes avec des valeurs manquantes
    cleaned_data = data.dropna()
    return cleaned_data

def split_data(data, test_size=0.2):
    """Divise les données en ensembles d'entraînement et de test."""
    X_train,X_test,y_train,y_test = train_test_split(data, test_size=test_size, random_state=42)
    return X_train,X_test,y_train,y_test

def save_transformed_data(data, filename):
    """Enregistre les données transformées."""
    data_file_path = Path('data/processed') / filename
    data.to_csv(data_file_path)

def load_transformed_data(filename):
    """Charge les données transformées."""
    data_file_path = Path('data/processed') / filename
    return pd.read_csv(data_file_path, index_col=0)
