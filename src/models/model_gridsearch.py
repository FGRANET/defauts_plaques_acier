from sklearn.model_selection import GridSearchCV
import joblib

class GridSearch:
    def __init__(self, estimator, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1):
        """
        Initialisation de la recherche par grille.
        
        :param estimator: Le modèle/estimateur sur lequel effectuer la recherche par grille.
        :param param_grid: Grille de paramètres à tester.
        :param cv: Nombre de folds de validation croisée.
        :param scoring: Métrique d'évaluation.
        :param verbose: Niveau de verbosité.
        :param n_jobs: Nombre de cœurs à utiliser pour le calcul parallèle.
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.grid_search = None

    def fit(self, X, y):
        """
        Exécute la recherche par grille sur les données fournies.
        
        :param X: Les caractéristiques d'entraînement.
        :param y: Les étiquettes cibles.
        """
        self.grid_search = GridSearchCV(estimator=self.estimator, param_grid=self.param_grid,
                                        cv=self.cv, scoring=self.scoring, verbose=self.verbose, n_jobs=self.n_jobs)
        self.grid_search.fit(X, y)
        print(f"Meilleurs paramètres: {self.grid_search.best_params_}")
        print(f"Meilleur score: {self.grid_search.best_score_}")

    def predict(self, X):
        """
        Fait des prédictions avec le meilleur modèle trouvé.
        
        :param X: Les caractéristiques pour lesquelles faire des prédictions.
        :return: Les prédictions du modèle.
        """
        if self.grid_search is None:
            raise Exception("GridSearchCV n'a pas encore été ajusté.")
        return self.grid_search.predict(X)

    def save_best_model(self, file_path):
        """
        Sauvegarde le meilleur modèle trouvé sur le disque.
        
        :param file_path: Chemin du fichier où sauvegarder le modèle.
        """
        if self.grid_search is None:
            raise Exception("GridSearchCV n'a pas encore été ajusté.")
        joblib.dump(self.grid_search.best_estimator_, file_path)
