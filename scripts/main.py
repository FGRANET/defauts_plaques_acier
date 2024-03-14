import argparse
from pathlib import Path
from src.data.load_data import load_csv_data
from src.data.clean_data import dropna, drop_duplicates
from src.data.split_data import split_data
from src.data.preprocess_data import standardize_data
from src.features.feature_engineering import select_k_best_features


parser = argparse.ArgumentParser(description="Pipeline d'exécution pour le projet DEFAUTS_PLAQUES_ACIER")
parser.add_argument('--load-data', action='store_true', help="Charge les données depuis le fichier CSV")
parser.add_argument('--clean-data', action='store_true', help="Nettoie les données chargés (doublons, valeurs manquantes)")
parser.add_argument('--split-data',action='store_true', help="Séparation du train en train et test")
parser.add_argument('--preprocess-data',action='store_true', help="Standardisation ou équivalent")
parser.add_argument('--feature-engineering',action='store_true', help="Création features / Réduction de dimension")

# Ajoutez d'autres arguments selon les besoins
args = parser.parse_args()


def main():
    if args.load_data:
        data_folder_path= "C:/Users/franc/DATA/DATA_Projet/Kaggle/defauts_plaques_acier"
        df = load_csv_data('train.csv', data_folder_path)
    
    if args.clean_data:
        df = dropna(df)
        df = drop_duplicates(df)

    if args.split_data:
        """" Séparation du df chargé et nettoyé en train et test. Cela suppose connaitre les features et la target"""
        data = df.iloc[:,:-7]
        target = df.iloc[:,-7:]
        X_train,X_test,y_train,y_test=split_data(data,target, test_size=0.2)
    
    if args.preprocess_data:
        X_train,X_test= standardize_data(X_train,X_test)
    
    if args.feature_engineering:
        X_train,X_test = select_k_best_features(X_train, y_train, X_test, k=10)







if __name__ == "__main__":
    main()
