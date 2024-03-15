from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def select_features_kbest(X_train, y_train, X_test, k=10):
    """
    Sélectionne les k meilleures caractéristiques.
    """
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    print(f"les {k} features les plus pertinentes ont été sélectionnées.")

    return X_train_selected, X_test_selected


def select_features_select_from_model(X_train,X_test, y_train, model=None, threshold='mean'):
    """
    Sélectionne les caractéristiques en utilisant SelectFromModel.

    :param X_train: Features d'entrainement.
    :param y_train: Target d'entraînement.
    :param model: Modèle à utiliser pour l'importance des caractéristiques. Si None, RandomForestClassifier est utilisé.
    :param threshold: Seuil pour la sélection des caractéristiques ('mean', 'median', ou une valeur flottante).
    :return: X_train avec les caractéristiques sélectionnées, X_test avec les caractéristiques sélectionnées et  liste des noms des caractéristiques sélectionnées.
    """
    if model is None:
        model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    selector = SelectFromModel(model, threshold=threshold, prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    selected_features = X_train.columns[selector.get_support()]
    
    return X_train_selected,X_test_selected, selected_features


def select_features_rfe(X_train,X_test, y_train, n_features_to_select=10, model=None):
    """
    Sélectionne les k meilleures caractéristiques en utilisant RFE.

    :param X_train: Features d'entrainement.
    :param y_train: Target d'entraînement.
    :param n_features_to_select: Nombre de caractéristiques à sélectionner.
    :param model: Modèle à utiliser pour l'évaluation des caractéristiques. Si None, RandomForestClassifier est utilisé.
    :return: X_train avec les caractéristiques sélectionnées, X_test avec les caractéristiques sélectionnées et  liste des noms des caractéristiques sélectionnées.
    """
    if model is None:
        model = RandomForestClassifier()
        
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe.fit(X_train, y_train)
    
    X_train_selected = rfe.transform(X_train)
    X_test_selected = rfe.transform(X_test)
    selected_features = X_train.columns[rfe.support_]
    
    return X_train_selected,X_test_selected, selected_features