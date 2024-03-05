from sklearn.metrics import roc_auc_score

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
        for column in self.true_labels.columns:
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


