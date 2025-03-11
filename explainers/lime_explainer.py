import numpy as np
from lime import lime_tabular
import matplotlib.pyplot as plt

def expliquer_lime(modele, entree, fond, noms_features, mode):
    explainer = lime_tabular.LimeTabularExplainer(training_data=fond, feature_names=noms_features, discretize_continuous=True, mode=mode)
    def pred(data):
        res = modele.predict(data)
        return res.ravel()
    exp = explainer.explain_instance(entree, pred if mode == 'regression' else modele.predict, num_features=len(noms_features) if noms_features else 5, top_labels=1)
    fig_lime = exp.as_pyplot_figure()
    fig_lime.set_size_inches(6, 3)
    return fig_lime, exp
