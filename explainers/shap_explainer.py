import shap
from streamlit_shap import st_shap

def expliquer_shap(modele, entree, shap_vals, explainer):
    val_shap = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
    val_shap = val_shap[0, :, 0]
    entree_2d = entree.reshape(1, -1)
    fig_force = shap.force_plot(explainer.expected_value[0], val_shap, entree_2d, matplotlib=False)
    st_shap(fig_force)
