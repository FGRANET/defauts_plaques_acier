import argparse
from pathlib import Path
from src.data.load_data import load_csv_data
from src.data.clean_data import dropna, drop_duplicates

parser = argparse.ArgumentParser(description="Pipeline d'exécution pour le projet DEFAUTS_PLAQUES_ACIER")
parser.add_argument('--load-data', action='store_true', help="Charge les données depuis le fichier CSV")
parser.add_argument('--clean-data', action='store_true', help="Nettoie les données chargés (doublons, valeurs manquantes)")

# Ajoutez d'autres arguments selon les besoins
args = parser.parse_args()


def main():
    if args.load_data:
        data_folder_path= "C:/Users/franc/DATA/DATA_Projet/Kaggle/defauts_plaques_acier"
        df = load_csv_data('train.csv', data_folder_path)
        print(df)
    
    if args.clean_data:
        df = dropna(df)
        df = drop_duplicates(df)

if __name__ == "__main__":
    main()
