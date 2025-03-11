import numpy as np
import plotly.graph_objects as go

def dessiner_reseau(tailles, poids, types, activations=None):
    fig = go.Figure()
    espacement = 1.0 / (len(tailles) - 1)
    positions = []
    for i, (nb, typ) in enumerate(zip(tailles, types)):
        x = i * espacement
        if activations is not None and i > 0 and (i - 1) < len(activations):
            nb_neurones = activations[i - 1].shape[1]
        else:
            nb_neurones = nb
        pos_couche = []
        for j in range(nb_neurones):
            y = (j + 0.5) / nb_neurones
            if activations is not None and i > 0 and (i - 1) < len(activations):
                val = activations[i - 1][0, j] if j < activations[i - 1].shape[1] else 0
            else:
                val = 0
            intensite = min(abs(val), 1.0)
            couleur = f"rgba(255,0,0,{intensite:.6f})" if val > 0 else "rgba(255,255,255,1)"
            texte = f"Couche {i+1} ({typ})<br>Neurone {j+1}<br>Activation: {val:.4f}"
            fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', marker=dict(size=20, color=couleur, line=dict(color='black', width=2)), hoverinfo="text", text=[texte]))
            pos_couche.append((x, y))
        positions.append(pos_couche)
    for i, matrice in enumerate(poids):
        maxi = np.max(np.abs(matrice)) if np.max(np.abs(matrice)) > 0 else 1
        for j, depart in enumerate(positions[i]):
            if j >= matrice.shape[0]:
                continue
            for k, arrivee in enumerate(positions[i+1]):
                if k >= matrice.shape[1]:
                    continue
                val_p = matrice[j, k]
                intensite_p = abs(val_p) / maxi
                intensite_p = max(0.0, min(intensite_p, 1.0))
                couleur_p = f"rgba(0,0,255,{intensite_p:.6f})"
                texte_lien = f"Poids: {val_p:.4f}<br>De Couche {i+1} Neurone {j+1} → Couche {i+2} Neurone {k+1}"
                fig.add_trace(go.Scatter(x=[depart[0], arrivee[0]], y=[depart[1], arrivee[1]], mode='lines', line=dict(color=couleur_p, width=2 + 4 * intensite_p), hoverinfo="text", text=[texte_lien]))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
    return fig

def dessiner_heatmaps(activations):
    figs = []
    for i, act in enumerate(activations):
        fig = go.Figure(data=go.Heatmap(z=act.T, colorscale="Viridis"))
        fig.update_layout(title=f"Activation Layer {i+1}", xaxis_title="Neurones", yaxis_title="Activation")
        figs.append(fig)
    return figs
