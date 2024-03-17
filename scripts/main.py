import argparse
import pandas as pd
from src.data.load_data import load_csv_data
from src.data.clean_data import dropna, drop_duplicates
from src.data.split_data import split_data
from src.data.preprocess_data import standardize_data
from src.features.select_features import select_features_kbest,select_features_select_from_model,select_features_rfe
from src.models.model_random_forest import* 
from src.evaluation.model_evaluation import*
import mlflow


mlflow.set_experiment("Defauts_Plques_Acier_Experiment")

parser = argparse.ArgumentParser(description="Pipeline d'exécution pour le projet DEFAUTS_PLAQUES_ACIER")
parser.add_argument('--load-data', action='store_true', help="Charge les données depuis le fichier CSV")
parser.add_argument('--clean-data', action='store_true', help="Nettoie les données chargés (doublons, valeurs manquantes)")
parser.add_argument('--split-data',action='store_true', help="Séparation du train en train et test")
parser.add_argument('--preprocess-data',action='store_true', help="Standardisation ou équivalent")
parser.add_argument('--feature-engineering',action='store_true', help="Création features")
parser.add_argument('--select-features', action='store_true', help="Activer la sélection de caractéristiques")
parser.add_argument('--method', type=str, default='select_from_model', choices=['select_kbest','select_from_model', 'rfe'], help="Méthode de sélection de caractéristiques à utiliser")
parser.add_argument('--model-random-forest', action='store_true', help="Entrainement random forest")
parser.add_argument('--model-evaluation', action='store_true', help="Evaluation sur le train et le test")
# Ajout d'autres arguments selon les besoins
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

        X_train,X_test,preprocessing_step= standardize_data(X_train,X_test)
        mlflow.log_param("preprocess", preprocessing_step)
    
    if args.select_features:
        if args.method =='select_kbest':
            X_train, X_test,selector = select_features_kbest(X_train, y_train, X_test, k=10) 
        elif args.method == 'select_from_model':
            X_train, X_test, selected_features,selector = select_features_select_from_model(X_train,X_test, y_train, model=None, threshold='mean')
            print(f"Caractéristiques sélectionnées : {selected_features}")
        elif args.method == 'rfe':
            X_train, X_test, selected_features,selector = select_features_rfe(X_train,X_test, y_train, n_features_to_select=10, model=None)
            print(f"Caractéristiques sélectionnées : {selected_features}")
        mlflow.log_param("selector", selector)
        
    
    if args.model_random_forest:
        model = RandomForestModel()
        model.train(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test))
        y_pred.columns = y_test.columns
        mlflow.sklearn.log_model(model.model, "Model")
        params = model.model.get_params()
        for param, value in params.items():
            mlflow.log_param(param, value)

    
    if args.model_evaluation:
        eval_test = ModelEvaluation(y_test, y_pred)
        score = eval_test.average_auc()
        print(f"La moyenne des scores AUC est:{score}")
        mlflow.log_metric("average_auc", score)


if __name__ == "__main__":
    main()
