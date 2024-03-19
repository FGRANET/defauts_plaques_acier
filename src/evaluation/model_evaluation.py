from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

class ModelEvaluation:
    def __init__(self, true_labels, predictions):
        """
        Initialisation de la classe d'évaluation.
        :param true_labels: les étiquettes réelles.
        :param predictions: les probabilités prédites par le modèle.
        """
        self.true_labels = true_labels
        self.predictions = predictions

    def calculate_auc(self):
        """
        Calculer l'AUC pour chaque catégorie de défaut.
        :return: dictionnaire des AUC pour chaque défaut.
        """
        auc_scores = {}
        for i,column in enumerate(self.true_labels.columns):
            auc = roc_auc_score(self.true_labels[column], self.predictions[column])
            auc_scores[column] = auc
        return auc_scores

    def average_auc(self):
        """
        Calculer la moyenne des scores AUC sur toutes les catégories de défaut.
        :return: moyenne des AUC.
        """
        auc_scores = self.calculate_auc()
        average_auc = sum(auc_scores.values()) / len(auc_scores)
        return average_auc


from sklearn.metrics import make_scorer

def adapt_pred_proba(model,X,y_true):
    y_pred_proba = model.predict_proba(X)
    prob_positives = np.array([proba[:, 1] for proba in y_pred_proba]).T  # Transposer pour avoir la forme correcte (n_samples, n_targets)
    y_pred_proba = pd.DataFrame(prob_positives, columns=y_true.columns)
    return y_pred_proba

def auc_scorer(y_true,y_pred_proba):
    """
    Fonction de scoring personnalisée pour calculer l'AUC moyen.
    
    :param model: le modèle à évaluer
    :param X: les données sur lesquelles faire des prédictions
    :param y_true: les vraies étiquettes
    """
    

    eval_model = ModelEvaluation(y_true, y_pred_proba)
    score = eval_model.average_auc()
    return score

# Créer un scorer scikit-learn à partir de la fonction de scoring personnalisée
average_auc_scorer = make_scorer(auc_scorer, needs_proba=True)