import streamlit as st
import numpy as np
from core.model_manager import charger_modele, extraire_infos, calculer_activations, calculer_shap
from core.graph_utils import dessiner_reseau, dessiner_heatmaps
from explainers.shap_explainer import expliquer_shap
from explainers.lime_explainer import expliquer_lime

st.set_page_config(page_title="Visualiseur NN", layout="wide")
st.title("Comprendre ton IA")

with st.sidebar:
    st.subheader("1) Charger le modèle")
    fichier_modele = st.file_uploader("Fichier .h5", type=["h5"])
    if fichier_modele is not None:
        try:
            modele = charger_modele(fichier_modele)
            st.session_state["modele"] = modele
            st.success("Modèle chargé !")
        except RuntimeError as erreur:
            st.error(str(erreur))
    if "modele" in st.session_state:
        dim_entree = st.session_state["modele"].input_shape[-1]
        st.subheader("2) Saisir les features")
        saisie = st.text_input(f"Entrez {dim_entree} valeurs (ex: 0.5, 0.2, -0.1, ...)", "1, 85, 66, 29, 0.26, 0.351, 31, 1")
        if st.button("Simuler"):
            try:
                valeurs = [float(x) for x in saisie.split(",")]
                if len(valeurs) != dim_entree:
                    st.error(f"Il faut {dim_entree} valeurs, vous en avez donné {len(valeurs)}.")
                else:
                    entree = np.array(valeurs)
                    activations = calculer_activations(st.session_state["modele"], entree)
                    st.session_state["activations"] = activations
                    prediction = st.session_state["modele"].predict(entree.reshape(1, -1))
                    st.session_state["prediction"] = prediction[0][0]
                    shap_vals, explainer = calculer_shap(st.session_state["modele"], entree)
                    st.session_state["shap_vals"] = shap_vals
                    st.session_state["explainer"] = explainer
                    st.session_state["entree"] = entree
                    st.success("Simulation effectuée !")
            except ValueError:
                st.error("Format invalide : entrez des nombres séparés par des virgules.")

if "modele" not in st.session_state:
    st.info("Veuillez charger un modèle dans la barre latérale.")
else:
    tab_reseau, tab_explain, tab_activation = st.tabs(["Réseau", "Explications", "Activations"])
    with tab_reseau:
        st.subheader("Visualisation du réseau")
        tailles, poids, types = extraire_infos(st.session_state["modele"])
        if "activations" in st.session_state:
            fig_reseau = dessiner_reseau(tailles, poids, types, st.session_state["activations"])
        else:
            fig_reseau = dessiner_reseau(tailles, poids, types)
        st.plotly_chart(fig_reseau, use_container_width=True)
        if "prediction" in st.session_state:
            st.write(f"**Votre prédiction est** : {st.session_state['prediction']}")
    with tab_explain:
        st.subheader("Interprétations")
        if "entree" not in st.session_state:
            st.info("Veuillez lancer la simulation dans la barre latérale.")
        else:
            methode = st.selectbox("Méthode d'explication", ["SHAP", "LIME"], index=0)
            if st.button("Expliquer"):
                if methode == "SHAP":
                    expliquer_shap(st.session_state["modele"], st.session_state["entree"], st.session_state["shap_vals"], st.session_state["explainer"])
                else:
                    dim = st.session_state["modele"].input_shape[-1]
                    fond = np.random.normal(0, 1, size=(100, dim))
                    noms = [f"Feature_{i}" for i in range(dim)]
                    fig_lime, _ = expliquer_lime(st.session_state["modele"], st.session_state["entree"], fond, noms, 'regression')
                    st.pyplot(fig_lime)
    with tab_activation:
        st.subheader("Heatmaps d'activations")
        if "activations" not in st.session_state:
            st.info("Veuillez lancer la simulation dans la barre latérale.")
        else:
            figs = dessiner_heatmaps(st.session_state["activations"])
            for f in figs:
                st.plotly_chart(f, use_container_width=True)
