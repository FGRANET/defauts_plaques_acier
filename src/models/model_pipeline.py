from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from data.load_data import load_csv_data
from features.feature_engineering import select_k_best_features

# Créer un pipeline
pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('classification', RandomForestClassifier())
])

# Vous pouvez également inclure votre propre transformation en tant qu'étape dans le pipeline,
# en suivant les conventions de scikit-learn pour créer un transformateur personnalisé.
