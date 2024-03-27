
import argparse
import pandas as pd
import numpy as np
import mlflow
import runpy

def find_project_directory():
    """
    Détermine le repertoire du projet
    """
    # Récupérer le chemin d'accès du répertoire courant
    current_dir = os.getcwd()
    # Accéder au répertoire parent en utilisant os.pardir : on accède alors au repertoire scr
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    
    return parent_dir

project_directory= find_project_directory()

import sys
sys.path.append(project_directory)

from src.data.load_data import load_csv_data

"""
from src.data.clean_data import dropna, drop_duplicates
from src.data.split_data import split_data
from src.data.preprocess_data import standardize_data
from src.features.select_features import select_features_kbest,select_features_select_from_model,select_features_rfe
from src.models.model_random_forest import* 
from src.models.model_gridsearch import*
from src.models.model_RandomizedSearchCV import*
from src.evaluation.model_evaluation import*
from src.models.multi_label import*
from src.evaluation.ml_flow import*
import runpy
"""

mlflow.set_experiment("Defauts_Plaques_Acier_Experiment")

parser = argparse.ArgumentParser(description="Pipeline d'exécution pour le projet DEFAUTS_PLAQUES_ACIER")
parser.add_argument('--load_select_data', action='store_true', help="Charge les données depuis le fichier CSV et selectionne les caractéristiques")

args = parser.parse_args()

def main():
    with mlflow.start_run():
        if args.load_select_data:
            #chargement des données
            runpy.run_path(dossier_github_frederic+'/src/features/select_features.py', init_globals=globals(), run_name='__main__')
            
            #chargement df
            df= load_csv_data(data_filename="train.csv",data_folder_path=dossier_github_frederic)
            #Séparation des variables et des cibles
            targets = df.iloc[:,-7:]
            #Chargement du DataFrame
            features =load_csv_data(data_filename="selected_features.csv",data_folder_path=dossier_github_frederic)
            number_features_eliminated = 27 - len(features.columns)
            print("Nombre de caractéristiques supprimées : " + str(number_features_eliminated))
            print("Caractéristiques conservées:",features.columns)
            mlflow.log_param("Nombre cacatéristiques supprimées", number_features_eliminated)


if __name__ == "__main__":
    main()