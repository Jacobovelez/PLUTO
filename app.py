import streamlit as st
import pandas as pd

from pluto_engine import (
    clean_projects_with_log,
    remove_outliers_iqr,
    kpis_global,
    kpis_by_complexity,
    plot_histograms,
    plot_boxplot_by_complexity,
    normality_test_by_complexity,
    plot_normality,
    fit_models_by_complexity,
    best_model_per_complexity,
    plot_models_overlay,
    completion_probability,
    get_best_model_for_complexity,
    evaluate_student_choice,
    fit_duration_linear_models,
    predict_duration,
    optimize_allocation,
    success_score,
)

st.set_page_config(page_title="PLUTO – Duration (mois)", layout="wide")

st.title("PLUTO – Module formatif : analyse de la durée (mois)")
st.caption("V2 : transparence nettoyage (NA + outliers) → KPIs → distributions.")

st.sidebar.header("1) Charger les données")
uploaded = st.sidebar.file_uploader("Fichier Excel (.xlsx) ou CSV", type=["xlsx", "xls", "csv"])

if uploaded is None:
    st.info("Charge un fichier pour commencer (ex: Base de données pluto.xlsx).")
    st.stop()

try:
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(uploaded)
    else:
        raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Impossible de lire le fichier: {e}")
    st.stop()

st.subheader("Aperçu des données brutes")
st.dataframe(raw.head(20), use_container_width=True)
st.write(f"Dimensions: **{raw.shape[0]} lignes × {raw.shape[1]} colonnes**")

st.sidebar.header("2) Nettoyage / Outliers")
st.sidebar.markdown("**Règles appliquées**")
st.sidebar.markdown("- Suppression des lignes avec données manquantes (NA) sur les colonnes clés")
use_outliers = st.sidebar.checkbox("Supprimer les outliers (IQR par complexité)", value=True)
iqr_k = st.sidebar.slider("Facteur IQR (k)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
st.sidebar.info(
    "🧮 **Comment fonctionne le facteur IQR (outliers)**\n\n"
    "On identifie les valeurs extrêmes de *Duration* avec la règle IQR, calculée **par niveau de complexité**.\n\n"
    "**Définitions :**\n"
    "- Q1 = 25e percentile\n"
    "- Q3 = 75e percentile\n"
    "- IQR = Q3 − Q1\n\n"
    "**Bornes :**\n"
    "- Borne basse = Q1 − k × IQR\n"
    "- Borne haute = Q3 + k × IQR\n\n"
    "Une durée est considérée comme **extrême** si elle est en dehors de ces bornes.\n\n"
    "Le paramètre **k** (par défaut 1.5) contrôle la sévérité du filtrage :\n"
    "- k plus petit → plus d’outliers supprimés\n"
    "- k plus grand → filtrage plus tolérant\n\n"
    "Objectif : obtenir une base **représentative des projets standards**, sans supprimer des projets “faux”, "
    "mais en limitant l’influence des cas très atypiques sur les statistiques et les modèles."
)

clean_df, na_removed, na_summary = clean_projects_with_log(raw)

st.subheader("Étape 1A — Nettoyage : suppression des valeurs manquantes (NA)")
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.metric("Lignes initiales", int(na_summary["n_initial"]))
with c2:
    st.metric("Lignes supprimées (NA)", int(na_summary["n_removed_na"]))
with c3:
    st.metric("% supprimé (NA)", f'{na_summary["pct_removed_na"]:.2f}%')

st.caption(
    "Les lignes avec NA sur les colonnes clés ne permettent pas de calculer correctement "
    "les indicateurs, ni d’ajuster un modèle prédictif."
)

st.subheader("Liste des lignes supprimées (NA)")
if len(na_removed) == 0:
    st.write("Aucune ligne supprimée pour NA.")
else:
    st.dataframe(na_removed, use_container_width=True)

st.subheader("Données après nettoyage NA (base de travail)")
st.dataframe(clean_df.head(20), use_container_width=True)
st.write(f"Dimensions après NA: **{clean_df.shape[0]} lignes × {clean_df.shape[1]} colonnes**")

st.subheader("Étape 1B — Valeurs extrêmes (outliers) sur Duration")
st.caption("Objectif : obtenir une base représentative des projets standards (sans cas extrêmes rares).")

work_df = clean_df
out_info = None
out_removed = None

if use_outliers:
    work_df, out_info, out_removed = remove_outliers_iqr(work_df, var="duration_months", group="complexity", k=iqr_k)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.metric("Lignes avant outliers", int(clean_df.shape[0]))
    with c2:
        st.metric("Outliers supprimés", int(out_info["outliers"].sum()) if out_info is not None else 0)
    with c3:
        pct = 100 * (out_info["outliers"].sum() / max(1, clean_df.shape[0])) if out_info is not None else 0
        st.metric("% supprimé (outliers)", f"{pct:.2f}%")

    st.subheader("Résumé outliers (bornes IQR par complexité)")
    st.dataframe(out_info, use_container_width=True)

    st.subheader("Liste des lignes supprimées (outliers)")
    if out_removed is None or len(out_removed) == 0:
        st.write("Aucun outlier supprimé.")
    else:
        st.dataframe(out_removed, use_container_width=True)
else:
    st.info("Suppression d'outliers désactivée (tu peux l'activer dans la barre latérale).")

st.divider()

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("KPIs globaux (Duration)")
    st.table(kpis_global(work_df, var="duration_months"))
with col2:
    st.subheader("KPIs par complexité (Duration)")
    st.dataframe(kpis_by_complexity(work_df, var="duration_months", group="complexity"), use_container_width=True)

st.subheader("Distributions (Duration) – global & par complexité")
fig = plot_histograms(work_df, var="duration_months", group="complexity")
st.pyplot(fig, clear_figure=True)
st.success("Nettoyage terminé ✅  Prochaine étape : Étape 2 (analyse segmentée) puis Étape 3 (test de normalité).")

st.divider()
st.header("Étape 2 — Analyse segmentée (par complexité)")

st.caption(
    "On compare les durées par niveau de complexité. "
    "Si les distributions diffèrent, on justifie une modélisation séparée par complexité."
)

fig_box = plot_boxplot_by_complexity(work_df, var="duration_months", group="complexity")
st.pyplot(fig_box, clear_figure=True)




st.divider()
st.header("Étape 3 — Test de normalité (par complexité)")

st.caption(
    "On teste si la durée suit une distribution normale pour chaque niveau de complexité. "
    "Ce test sert à orienter le choix du modèle statistique."
)

norm_table = normality_test_by_complexity(
    work_df,
    var="duration_months",
    group="complexity"
)

st.subheader("Résultats du test de normalité (Shapiro–Wilk)")
st.dataframe(norm_table, use_container_width=True)

st.subheader("Visualisation : histogramme + loi normale")
fig_norm = plot_normality(
    work_df,
    var="duration_months",
    group="complexity"
)
st.pyplot(fig_norm, clear_figure=True)

st.divider()
st.header("Étape 3.5 — Choix de la distribution (par complexité)")

st.caption(
    "Avant que PLUTO révèle la meilleure distribution, tu choisis toi-même une loi pour chaque complexité. "
    "Ensuite PLUTO compare ton choix au meilleur modèle selon le critère BIC et explique pourquoi."
)

# On calcule les ajustements une seule fois (on les réutilise ensuite pour l’étape 4)
fit_tbl = fit_models_by_complexity(work_df, var="duration_months", group="complexity")

# PLUTO : meilleur modèle par complexité selon BIC
best_bic = best_model_per_complexity(fit_tbl, criterion="bic")

models_list = ["normal", "lognorm", "gamma", "weibull", "expon"]
complexities = sorted(work_df["complexity"].unique().tolist())

st.subheader("Choix étudiant")
student_choices = {}

for comp in complexities:
    # choix de l'étudiant
    student_choices[comp] = st.selectbox(
        f"Choisis une distribution pour la complexité : {comp}",
        models_list,
        key=f"choice_{comp}"
    )

st.subheader("Feedback PLUTO (comparaison BIC)")
for comp in complexities:
    chosen = student_choices[comp]
    res = evaluate_student_choice(fit_tbl, complexity_value=comp, chosen_model=chosen, criterion="bic")

    st.markdown(f"### Complexité : `{comp}`")
    st.write(f"**Ton choix :** {chosen}")
    st.write(f"**PLUTO (meilleur BIC) :** {res.get('best_model', '—')}")
    st.write(f"**Verdict :** {res['status']}")
    st.info(res["message"])
    if "extra" in res:
        st.caption(res["extra"])

    # optionnel : afficher le graphique des modèles pour aider visuellement
    with st.expander("Voir les distributions ajustées (histogramme + modèles)"):
        fig_overlay = plot_models_overlay(
            work_df,
            var="duration_months",
            group="complexity",
            fit_table=fit_tbl,
            complexity_value=comp
        )
        if fig_overlay is not None:
            st.pyplot(fig_overlay, clear_figure=True)
        else:
            st.warning("Pas assez de données pour tracer les modèles.")

st.divider()
st.header("Étape 4 — Sélection du modèle statistique optimal (par complexité)")

st.caption(
    "On teste plusieurs modèles statistiques pour la durée (Gaussian, Log-Normal, Gamma, Weibull, Exponential). "
    "On compare leurs performances avec AIC/BIC et on sélectionne le meilleur modèle par complexité."
)


st.subheader("Tableau comparatif des modèles (AIC / BIC)")
st.dataframe(fit_tbl.sort_values(["complexity", "aic"]), use_container_width=True)

best_bic = best_model_per_complexity(fit_tbl, criterion="bic")
st.subheader("Meilleur modèle par complexité (critère BIC)")
st.dataframe(best_bic[["complexity", "model", "n", "aic", "bic"]], use_container_width=True)


st.subheader("Visualisation : histogramme + modèles ajustés (par complexité)")
complexities = sorted(work_df["complexity"].unique().tolist())
selected_comp = st.selectbox("Choisir une complexité", complexities)

fig_overlay = plot_models_overlay(
    work_df,
    var="duration_months",
    group="complexity",
    fit_table=fit_tbl,
    complexity_value=selected_comp
)
if fig_overlay is not None:
    st.pyplot(fig_overlay, clear_figure=True)
else:
    st.warning("Pas assez de données pour tracer les modèles.")

st.divider()
st.header("Étape 5 — Simulateur 'What-if' (curseurs) + solution optimale PLUTO")

st.caption(
    "Ici, on simule l’impact des leviers (effort, coût, révisions) sur la durée. "
    "L’objectif est pédagogique : tester des scénarios et comparer à une solution optimale calculée par PLUTO "
    "(minimisation de la durée sous contraintes)."
)

# 1) On entraîne des modèles linéaires par complexité (sur la base nettoyée)
models = fit_duration_linear_models(work_df, group="complexity")

complexities = sorted(work_df["complexity"].unique().tolist())
comp = st.selectbox("Complexité", complexities, key="step5_comp")

if comp not in models:
    st.warning("Pas assez de données pour ajuster un modèle what-if sur cette complexité.")
    st.stop()

model = models[comp]

# 2) Définir une zone 'standard' (bornes par défaut) à partir des quantiles
sub = work_df[work_df["complexity"] == comp].dropna(subset=["effort_my","cost_keur","revisions_number","duration_months"])
q = sub[["effort_my","cost_keur","revisions_number","duration_months"]].quantile([0.1, 0.5, 0.9])

eff_std = float(q.loc[0.5,"effort_my"])
cost_std = float(q.loc[0.5,"cost_keur"])
rev_std = float(q.loc[0.5,"revisions_number"])

st.subheader("Choix des curseurs (scénario étudiant)")
c1, c2, c3 = st.columns(3)

with c1:
    effort = st.slider("Effort (unités)", float(q.loc[0.1,"effort_my"]), float(q.loc[0.9,"effort_my"]), eff_std, step=0.5)
with c2:
    cost = st.slider("Coût (k€)", float(q.loc[0.1,"cost_keur"]), float(q.loc[0.9,"cost_keur"]), cost_std, step=50.0)
with c3:
    revisions = st.slider("Nb de révisions", int(q.loc[0.1,"revisions_number"]), int(q.loc[0.9,"revisions_number"]), int(rev_std), step=1)

student_duration = predict_duration(model, effort, cost, revisions)

st.markdown("### Résultat étudiant")
st.metric("Durée prédite (mois)", f"{student_duration:.1f}")

# 3) Contraintes pour l'optimisation PLUTO (l'étudiant peut les fixer)
st.subheader("Contraintes pour l’optimisation PLUTO")
st.caption("PLUTO cherche la meilleure combinaison (effort, coût, révisions) qui minimise la durée prédite, sous contraintes.")

colA, colB, colC = st.columns(3)
with colA:
    effort_max = st.slider("Effort max", float(q.loc[0.5,"effort_my"]), float(q.loc[0.9,"effort_my"]), float(q.loc[0.9,"effort_my"]), step=0.5)
with colB:
    cost_max = st.slider("Budget max (k€)", float(q.loc[0.5,"cost_keur"]), float(q.loc[0.9,"cost_keur"]), float(q.loc[0.9,"cost_keur"]), step=50.0)
with colC:
    rev_max = st.slider("Révisions max", int(q.loc[0.5,"revisions_number"]), int(q.loc[0.9,"revisions_number"]), int(q.loc[0.9,"revisions_number"]), step=1)

opt = optimize_allocation(
    model,
    effort_range=(float(q.loc[0.1,"effort_my"]), effort_max),
    cost_range=(float(q.loc[0.1,"cost_keur"]), cost_max),
    rev_range=(int(q.loc[0.1,"revisions_number"]), rev_max),
    step_effort=0.5,
    step_cost=50.0,
    step_rev=1
)

st.markdown("### Solution optimale PLUTO (min durée sous contraintes)")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Effort optimal", f"{opt['effort_my']:.1f}")
with c2:
    st.metric("Coût optimal (k€)", f"{opt['cost_keur']:.0f}")
with c3:
    st.metric("Révisions optimales", f"{opt['revisions_number']}")
with c4:
    st.metric("Durée optimale (mois)", f"{opt['duration_pred']:.1f}")

# 4) Score de succès
score = success_score(student_duration, opt["duration_pred"])
st.subheader("Pourcentage de succès")
st.metric("Score (0–100)", f"{score:.1f}%")

# 5) Explication claire
st.info(
    "Interprétation :\n"
    "- Ta durée prédite dépend des curseurs (effort, coût, révisions).\n"
    "- PLUTO calcule ensuite la combinaison qui minimise la durée, tout en respectant les contraintes.\n"
    "- Le score mesure à quel point ton scénario est proche de la performance optimale (en durée)."
)

st.caption(
    "Note méthodologique : ce simulateur est un outil pédagogique 'what-if' basé sur les tendances observées "
    "dans l'historique. Il ne prouve pas une causalité parfaite, mais permet de raisonner sur des compromis."
)
