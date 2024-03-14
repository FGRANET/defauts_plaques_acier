import mlflow
from sklearn.model_selection import train_test_split
from model_pipeline import pipeline
from ..data.load_data import load_csv_data


# Chargement et préparation des données
df = load_csv_data("train.csv")
features = df.iloc[:,:-8]
target = df.iloc[:,:-7]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# Initialisation de MLflow
mlflow.set_experiment('my_experiment')

with mlflow.start_run():
    # Configuration du modèle et des paramètres
    params = {'n_estimators': 100, 'max_depth': 4}
    pipeline.set_params(**params)

    # Entraînement et évaluation du modèle
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    # Logging des paramètres, métriques, et du modèle
    mlflow.log_params(params)
    mlflow.log_metric('accuracy', score)
    mlflow.sklearn.log_model(pipeline, 'model')
