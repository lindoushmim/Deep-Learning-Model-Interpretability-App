#  XAI - Intelligence Artificielle Explicable

Ce dépôt contient les ressources d'un projet de recherche sur l'**Intelligence Artificielle Explicable (XAI)**.

## Contenu du dépôt

- **Notebook d'expérimentation** : Tests de différentes méthodes de XAI dans un contexte éducatif.
- **Article scientifique** : Présentation des résultats et méthodologies explorées.
- **Interface interactive** : Application Streamlit pour visualiser et expliquer le fonctionnement des réseaux de neurones.


## Prérequis

Avant de commencer, assurez-vous d'avoir :

- **Python 3.7 ou version supérieure**
- Les bibliothèques suivantes installées :
  - `streamlit`
  - `tensorflow` / `keras`
  - `numpy`
  - `plotly`
  - `shap`
  - `lime`
  - `matplotlib`
- Un environnement virtuel est **fortement recommandé** (`venv` ou `conda`).



## Installation des dépendances

### Cloner le dépôt

- git clone https://forge.univ-lyon1.fr/p1805862/xai.git
- cd xai

Sur Mac/Linux : 
- python -m venv venv
- source venv/bin/activate

Sur Windows : 
- python -m venv venv
- venv\Scripts\activate

Installer les dépendances
- pip install nomDépendance


## Lancer l'interface Streamlit
Pour démarrer l'application Streamlit, exécutez :
streamlit run app.py

Si vous utilisez un environnement virtuel, exécutez : 
venv/bin/python -m streamlit run app.py

Si vous n'avez pas de modele de réseaux de neurone enregistré dans un fichier .h5, vous pouvez en trouver dans le dossier model pour tester l'application. 

## Expérimentations pour la recherche
Le notebook inclus dans ce dépôt contient l'ensemble du code pour nos tests. 
