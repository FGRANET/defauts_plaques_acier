from sklearn.feature_selection import SelectKBest, f_classif

def select_k_best_features(X_train, y_train, X_test, k=10):
    """
    Sélectionne les k meilleures caractéristiques.
    """
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    print(f"les {k} features les plus pertinentes ont été sélectionnées.")

    return X_train_selected, X_test_selected
