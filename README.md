

# XAI - Intelligence Artificielle Explicable

Visualisation et Interprétation de Réseaux de Neurones avec Streamlit

## 🔍 Aperçu du Projet

<img src="data/app" alt="image de l'app" width="500" height="500">


Ce projet propose une interface interactive construite avec **Streamlit** permettant de :

1. **Charger un modèle de réseau de neurones** au format `.h5`.
2. **Visualiser la structure du réseau de neurones**, avec une représentation où :
   - Les **liens entre les neurones** sont proportionnels aux poids.
   - Les **neurones activés** lors de la prédiction s'affichent en **rouge**.
3. **Saisir des données d'entrée** pour obtenir une prédiction du modèle.
4. **Interpréter la prédiction** grâce aux méthodes d'explication **SHAP** et **LIME**.

---

## 🔄 Installation et Lancement

### Prérequis

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


### Cloner le dépôt

- git clone https://github.com/lindoushmim/Deep-Learning-Model-Interpretability-App.git
- cd xai

### Créer un environnement

Sur **Mac/Linux** :  
- python -m venv venv  
- source venv/bin/activate  

Sur **Windows** :  
- python -m venv venv  
- venv\Scripts\activate  

### Installer les dépendances

- pip install -r requirements.txt  

### Lancer l'interface 

Pour démarrer l'application, exécutez :  
- venv/bin/python -m streamlit run app.py  

### Puis tester

Si vous n'avez pas de modèle de réseau de neurones enregistré dans un fichier `.h5`, vous pouvez en trouver dans le dossier **model** pour tester l'application.
