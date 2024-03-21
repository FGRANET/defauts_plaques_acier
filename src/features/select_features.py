from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif,mutual_info_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import collections
import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tabulate
import math
import collections


from sklearn.feature_selection import (
    SelectKBest, 
    chi2, 
    f_classif, 
    f_regression,
    r_regression,
    mutual_info_classif,
    mutual_info_regression
)

def load_raw_data(name):
    # Définir le chemin du dossier actuel du notebook
    current_dir = os.getcwd()
    
    # Accéder au répertoire parent
    parent_dir = os.path.dirname(current_dir)
    
    parent_dir_root = os.path.dirname(parent_dir)

    raw_data_dir = parent_dir_root  + "/data_to_use/raw"

    # Afficher le chemin du répertoire parent
    print("Répertoire des données brutes: \n", raw_data_dir)
    
    #Chargement d'un dataframe
    df=pd.read_csv(raw_data_dir + name, index_col=0)
        
    return df

# Méthodologie de sélection des features

def correlation_selection(df,features,targets,target,correlation_threshold=0.1):
    """
    Sélectionne les caractéristiques numériques parmi les feature du dataframe df
    qui ont une correlation avec la cible (target) supérieure au seuil de selection
    (threshold).
    """
    #Concaténation features et target
    concat = pd.concat([features,targets[target]],axis=1)
    
    #Matrice de corrélation
    correlations= concat.corr()[target]

    selected_features = correlations[abs(correlations) > correlation_threshold]

    # Retirer la cible
    remove_target = selected_features.index[selected_features.index != target]

    return selected_features[remove_target].index



def select_features_kbest(df, features, target, method, k=20):
    """
    Sélectionne à l'aide de la méthode KBest les k meilleures caractéristiques selon
    différentes méthodes statistiques, avec k=20 par défaut
    Retourne un dataframe contenant les k variables les plus pertinentes
    """
    # Sélectionner les k meilleures caractéristiques
    selector = SelectKBest(method, k=k)
    selected_features = selector.fit_transform(features, df[target])

    # Transformation en DataFrame
    selected_features = pd.DataFrame(selected_features)

    # Renommer les colonnes en utilisant les noms des colonnes sélectionnées
    selected_features.columns = selector.get_feature_names_out()

    # Sélectionner les colonnes sélectionnées
    selected_columns = selected_features.columns

    return df[selected_columns].columns


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
    
    return X_train_selected,X_test_selected, selected_features,"select_from_model"


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
    
    return X_train_selected,X_test_selected, selected_features,"rfe"


#Comptage des variables non utilisés

def count_useless_features(df,features,targets,method,correlation_threshold=0.1):
       
    # Créer un defaultdict avec une valeur initiale de 0
    dictionnaire = collections.defaultdict(int)
    """
    Pour chaque target :
        - sélectionne les caractéristiques numériques parmi les feature du dataframe df
    qui ont une correlation avec la cible (target) supérieure au seuil de selection
    (threshold).
        - Un dictionnaire est utilisé pour compter le nombre d'occurence de chaque variable
    non utilisée.
    Le dictionnaire final est trié par ordre décroissant
    """
    
    # Ajouter une entrée pour chaque clé de la liste avec une valeur de 0
    for cle in features:
        dictionnaire[cle] = 0

    for target in targets.columns:
        if method=="correlation":            

            selected_features = correlation_selection(df=df,
                                                    features=features,
                                                    targets=targets,target=target,
                                                    correlation_threshold=correlation_threshold)
        elif method == "f_classif": 

            selected_features=select_features_kbest(df=df,features=features,target=target,
                                                    method=f_classif,
                                                    k=20)
        elif method == "mutual_info_classif":

            selected_features=select_features_kbest(df=df,features=features,target=target,
                                                    method=mutual_info_classif,
                                                    k=20)    
        else:
            print("Vous n'avez pas choisi une méthode appropriée.")

        #Variables non selectionnées                
        non_used_features = [x for x in features if x not in selected_features]
            
        # Augmenter le compteur de 1 pour chaque variable non utilisée
        for cle in non_used_features:
                dictionnaire[cle] += 1            
     
    # Trier le dictionnaire par les valeurs
    dictionnaire_trie = sorted(dictionnaire.items(), key=lambda x: x[1],reverse=True)

    # Convertir la liste de tuples triés en dictionnaire
    dictionnaire_useless_features = dict(dictionnaire_trie)    

    return dictionnaire_useless_features


"""
def select_features_kbest(X_train, y_train, X_test, k=20):
    
    Sélectionne les k meilleures caractéristiques, avec k=20 par défaut
    
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    print(f"les {k} features les plus pertinentes ont été sélectionnées.")

    return X_train_selected, X_test_selected,"selectKbest"

"""

#Programme principal
while True:
    try:
        nombre = int(input("Entrez un nombre entier compris entre 1 et 27 pour selectionner autant de variables: "))
        if 1 <= nombre <= 27:
            print(f"Vous avez choisi de selectionner {nombre} caratéristiques.")
            break
        else:
            print("Votre nombre n'est pas compris entre 1 et 27. Veuillez réessayer.")
    except ValueError:
        print("Vous n'avez pas entré un nombre entier. Veuillez réessayer.")   

#Chargement du DataFrame
df= load_raw_data("/train.csv")
#Séparation des variables et des cibles
features = df.iloc[:,:-7]
targets = df.iloc[:,-7:]

while True:
    try:
        method = str(input("Quelle méthode voulez vous utiliser en premier pour commencer à compter les variables non utilisées : correlation,f_classif ou mutual_info_classif?"))
        if method in ["correlation","f_classif","mutual_info_classif"]:
            print(f"Vous avez choisi la méthode {method}. Patience, je calcule...",end="\n\n")
            break
        else:
            print("Vous n'avez pas choisi une méthode disponible. Veuillez réessayer.")
    except ValueError:
        print("Vous n'avez pas entré une chaîne de caractères. Veuillez réessayer.")

dictionnaire_useless_features = count_useless_features(df=df,features=features,targets=targets, method=method,correlation_threshold=0.1)
print(f"Voici les variables les moins utilisées par la méthode {method}] :", dictionnaire_useless_features, end="\n\n")

print("Je réalise les mêmes calculs pour les autres méthodes de sélection, patientez... ",end="\n")

dictionnaire_trie = dico_f_classif_trie=count_useless_features(df=df,features=features,targets=targets, method="correlation",correlation_threshold=0.1)
dico_f_classif_trie=count_useless_features(df=df,features=features,targets=targets, method="f_classif",correlation_threshold=0.1)
dico_mutual_info_trie = count_useless_features(df=df,features=features,targets=targets, method="mutual_info_classif",correlation_threshold=0.1) 

result = {}

for key in dictionnaire_trie:
    if key in dico_f_classif_trie:
        result[key] = dictionnaire_trie[key] + dico_f_classif_trie[key]
    else:
        result[key] = dictionnaire_trie[key]

for key in dico_f_classif_trie:
    if key not in dictionnaire_trie:
        result[key] = dico_f_classif_trie[key]

for key in dico_mutual_info_trie:
    if key in result:
        result[key] += dico_mutual_info_trie[key]
    else:
        result[key] = dico_mutual_info_trie[key]
print("\n\n Voici les variable les moins utilisées pour l'ensemble des trois méthodes sont : ", result)

