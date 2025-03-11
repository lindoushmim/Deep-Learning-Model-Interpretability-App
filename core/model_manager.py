import streamlit as st
import numpy as np
import tempfile
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.layers import Input, Dense

def charger_modele(fichier):
    if "nom_modele" in st.session_state and st.session_state["nom_modele"] != fichier.name:
        for cle in ["modele", "activations", "prediction", "shap_vals", "explainer", "entree"]:
            st.session_state.pop(cle, None)
    st.session_state["nom_modele"] = fichier.name
    fichier.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(fichier.read())
        chemin = tmp.name
    modele = load_model(chemin)
    if isinstance(modele, Sequential) and not modele.built:
        forme = modele.layers[0].input_shape
        if forme is None:
            raise ValueError("Le modèle n'a pas de shape d'entrée.")
        nouvelle_entree = Input(shape=forme[1:])
        sortie = modele(nouvelle_entree)
        modele = Model(nouvelle_entree, sortie)
    return modele

def extraire_infos(modele):
    tailles = [modele.input_shape[-1]]
    poids = []
    types = ["Input"]
    for couche in modele.layers:
        if isinstance(couche, Dense):
            tailles.append(couche.units)
            types.append("Hidden")
            w = couche.get_weights()
            if w:
                poids.append(w[0])
    types[-1] = "Output"
    return tailles, poids, types

def calculer_activations(modele, entree):
    if isinstance(modele, Sequential):
        entree_modele = modele.layers[0].input
    else:
        entree_modele = modele.input
    sorties = [c.output for c in modele.layers if isinstance(c, Dense)]
    modele_act = Model(entree_modele, sorties)
    act = modele_act.predict(np.array([entree]))
    return act

def calculer_shap(modele, entree, fond=None):
    if fond is None:
        nb = modele.input_shape[-1]
        fond = np.random.normal(0, 1, size=(10, nb))
    import shap
    explainer = shap.KernelExplainer(modele.predict, fond)
    entree_2d = entree.reshape(1, -1)
    shap_vals = explainer.shap_values(entree_2d, nsamples=100)
    return shap_vals, explainer
