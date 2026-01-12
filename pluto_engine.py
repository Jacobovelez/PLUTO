from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def clean_projects_with_log(raw: pd.DataFrame):
    """Nettoyage + transparence:
    - Standardise les colonnes
    - Log des lignes supprimées pour NA (sur colonnes clés)
    - Garde uniquement duration > 0
    Retourne (df_clean, df_removed_na, summary_dict).
    """
    df = raw.copy()

    rename_map = {
        "effort mxy": "effort_my",
        "duration  months": "duration_months",
        "revisions number": "revisions_number",
        "cost k€": "cost_keur",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    needed = ["complexity", "duration_months"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}. Colonnes disponibles: {list(df.columns)}")

    keep = [c for c in ["complexity", "duration_months", "effort_my", "revisions_number", "cost_keur"] if c in df.columns]
    df = df[keep].copy()

    df["complexity"] = (
        df["complexity"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    df["complexity"] = df["complexity"].replace({
        "1st_in_family": "first_in_family",
        "first_in_family": "first_in_family",
    })

    df["duration_months"] = pd.to_numeric(df["duration_months"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    key_cols = ["complexity", "duration_months"]
    mask_na = df[key_cols].isna().any(axis=1)

    df_removed_na = df.loc[mask_na].copy().reset_index(drop=True)
    df_clean = df.loc[~mask_na].copy()

    df_clean = df_clean[df_clean["duration_months"] > 0].copy()

    n_initial = int(df.shape[0])
    n_removed_na = int(df_removed_na.shape[0])
    pct = 100.0 * n_removed_na / max(1, n_initial)

    summary = {
        "n_initial": n_initial,
        "n_removed_na": n_removed_na,
        "pct_removed_na": pct,
        "key_cols_for_na": key_cols,
    }

    return df_clean.reset_index(drop=True), df_removed_na, summary


def remove_outliers_iqr(df: pd.DataFrame, var: str, group: str, k: float = 1.5):
    """Outliers IQR par groupe. Retourne (df_filtré, résumé, df_outliers)."""
    work = df.copy()
    rows = []
    keep_mask = np.ones(len(work), dtype=bool)
    outlier_rows = []

    for g, sub in work.groupby(group):
        x = sub[var].dropna().values
        if len(x) < 5:
            rows.append({"complexity": g, "total": int(len(sub)), "outliers": 0, "pct_outliers": 0.0,
                         "lower": np.nan, "upper": np.nan})
            continue

        q1 = np.quantile(x, 0.25)
        q3 = np.quantile(x, 0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr

        idx = sub.index
        is_out = (work.loc[idx, var] < lower) | (work.loc[idx, var] > upper)
        keep_mask[idx] = ~is_out

        if is_out.any():
            outlier_rows.append(work.loc[idx[is_out]].copy())

        out_n = int(is_out.sum())
        rows.append({
            "complexity": g,
            "total": int(len(sub)),
            "outliers": out_n,
            "pct_outliers": round(100 * out_n / max(1, len(sub)), 2),
            "lower": round(float(lower), 3),
            "upper": round(float(upper), 3),
        })

    info = pd.DataFrame(rows).sort_values("complexity").reset_index(drop=True)
    filtered = work.loc[keep_mask].reset_index(drop=True)
    out_df = pd.concat(outlier_rows, ignore_index=True) if len(outlier_rows) else pd.DataFrame(columns=work.columns)

    return filtered, info, out_df


def kpis_global(df: pd.DataFrame, var: str):
    s = df[var].dropna()
    return pd.DataFrame([{
        "n": int(s.shape[0]),
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
        "se": float(s.std(ddof=1) / np.sqrt(s.shape[0])) if s.shape[0] > 1 else 0.0,
    }]).round(3)


def kpis_by_complexity(df: pd.DataFrame, var: str, group: str):
    def _agg(s):
        n = s.shape[0]
        std = s.std(ddof=1) if n > 1 else 0.0
        return pd.Series({
            "n": int(n),
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(std),
            "se": float(std / np.sqrt(n)) if n > 1 else 0.0,
        })
    out = df.groupby(group)[var].apply(lambda x: _agg(x.dropna())).reset_index()
    return out.round(3)

def plot_histograms(df: pd.DataFrame, var: str, group: str):
    # On affiche d'abord les complexités, puis le global en dernier
    groups = sorted(df[group].dropna().unique().tolist())

    # +1 pour le graphique global
    n_plots = len(groups) + 1
    ncols = 2
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    # 1) Histogrammes par complexité
    for i, g in enumerate(groups):
        sub = df.loc[df[group] == g, var].dropna()
        axes[i].hist(sub, bins=20)
        axes[i].set_title(g)
        axes[i].set_xlabel(var)
        axes[i].set_ylabel("Count")

    # 2) Histogramme global (dernier)
    idx_global = len(groups)
    axes[idx_global].hist(df[var].dropna(), bins=20)
    axes[idx_global].set_title("Global")
    axes[idx_global].set_xlabel(var)
    axes[idx_global].set_ylabel("Count")

    # Masquer les axes inutilisés
    for j in range(idx_global + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    return fig

def plot_boxplot_by_complexity(df: pd.DataFrame, var: str, group: str):
    """Boxplot de var par groupe (complexity)."""
    work = df[[group, var]].dropna().copy()
    order = sorted(work[group].unique().tolist())

    fig, ax = plt.subplots(figsize=(10, 5))
    data = [work.loc[work[group] == g, var].values for g in order]

    ax.boxplot(data, labels=order, showfliers=True)
    ax.set_title(f"Boxplot de {var} par {group}")
    ax.set_xlabel(group)
    ax.set_ylabel(var)
    plt.xticks(rotation=15)
    fig.tight_layout()
    return fig
from scipy.stats import shapiro, norm

def normality_test_by_complexity(df, var: str, group: str):
    """Test de normalité (Shapiro) par groupe."""
    rows = []

    for g, sub in df.groupby(group):
        x = sub[var].dropna().values
        n = len(x)

        if n < 3:
            rows.append({
                "complexity": g,
                "n": n,
                "mean": None,
                "median": None,
                "std": None,
                "shapiro_p": None,
                "normality": "Non testable (n<3)"
            })
            continue

        mean = float(x.mean())
        median = float(np.median(x))
        std = float(x.std(ddof=1)) if n > 1 else 0.0

        try:
            pval = shapiro(x)[1]
        except Exception:
            pval = None

        normality = (
            "Normalité plausible"
            if (pval is not None and pval >= 0.05)
            else "Normalité non plausible"
        )

        rows.append({
            "complexity": g,
            "n": n,
            "mean": round(mean, 2),
            "median": round(median, 2),
            "std": round(std, 2),
            "shapiro_p": round(pval, 4) if pval is not None else None,
            "normality": normality
        })

    return pd.DataFrame(rows)

def plot_normality(df, var: str, group: str):
    """Histogramme + loi normale ajustée par complexité."""
    groups = sorted(df[group].unique())
    n = len(groups)

    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 4*n))
    if n == 1:
        axes = [axes]

    for ax, g in zip(axes, groups):
        x = df.loc[df[group] == g, var].dropna().values
        if len(x) < 3:
            ax.set_title(f"{g} (données insuffisantes)")
            continue

        mean, std = x.mean(), x.std(ddof=1)
        ax.hist(x, bins=20, density=True, alpha=0.6)
        xmin, xmax = ax.get_xlim()
        xs = np.linspace(xmin, xmax, 200)
        ax.plot(xs, norm.pdf(xs, mean, std), linewidth=2)
        ax.set_title(f"{g} — histogramme + loi normale")
        ax.set_xlabel(var)
        ax.set_ylabel("Densité")

    fig.tight_layout()
    return fig

from scipy import stats

def _safe_pos(x: np.ndarray) -> np.ndarray:
    """Assure des valeurs strictement positives (pour gamma/lognorm/weibull/expon)."""
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return x
    eps = np.min(x[x > 0]) / 1000 if np.any(x > 0) else 1e-6
    x = np.where(x <= 0, eps, x)
    return x

def fit_models_by_complexity(df: pd.DataFrame, var: str, group: str):
    """
    Ajuste plusieurs lois par groupe et retourne un tableau (AIC/BIC + params).
    """
    candidates = {
        "normal": stats.norm,
        "lognorm": stats.lognorm,
        "gamma": stats.gamma,
        "weibull": stats.weibull_min,
        "expon": stats.expon,
    }

    rows = []
    for g, sub in df.groupby(group):
        x_raw = sub[var].dropna().values
        n = len(x_raw)
        if n < 5:
            continue

        for name, dist in candidates.items():
            x = x_raw.copy()
            if name in ["lognorm", "gamma", "weibull", "expon"]:
                x = _safe_pos(x)

            # fit MLE (scipy retourne des paramètres: shape(s), loc, scale)
            try:
                params = dist.fit(x)
                loglik = np.sum(dist.logpdf(x, *params))
                k = len(params)
                aic = 2*k - 2*loglik
                bic = np.log(n)*k - 2*loglik

                rows.append({
                    "complexity": g,
                    "model": name,
                    "n": n,
                    "aic": aic,
                    "bic": bic,
                    "params": params,
                })
            except Exception:
                rows.append({
                    "complexity": g,
                    "model": name,
                    "n": n,
                    "aic": np.nan,
                    "bic": np.nan,
                    "params": None,
                })

    out = pd.DataFrame(rows)
    # Arrondi lisible
    if not out.empty:
        out["aic"] = out["aic"].round(2)
        out["bic"] = out["bic"].round(2)
    return out

def best_model_per_complexity(fit_table: pd.DataFrame, criterion: str = "aic"):
    """
    Renvoie le meilleur modèle par complexité selon criterion ('aic' ou 'bic').
    """
    if fit_table.empty:
        return fit_table
    t = fit_table.dropna(subset=[criterion]).copy()
    idx = t.groupby("complexity")[criterion].idxmin()
    best = t.loc[idx].sort_values("complexity").reset_index(drop=True)
    return best

def plot_models_overlay(df: pd.DataFrame, var: str, group: str, fit_table: pd.DataFrame, complexity_value: str):
    """
    Histogramme (density) + courbes des modèles ajustés pour une complexité donnée.
    """
    sub = df[df[group] == complexity_value].copy()
    x_raw = sub[var].dropna().values
    if len(x_raw) < 5:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    # histogramme density
    ax.hist(x_raw, bins=20, density=True, alpha=0.6)
    xmin, xmax = np.min(x_raw), np.max(x_raw)
    xs = np.linspace(xmin, xmax, 300)

    # modèles à tracer
    candidates = {
        "normal": stats.norm,
        "lognorm": stats.lognorm,
        "gamma": stats.gamma,
        "weibull": stats.weibull_min,
        "expon": stats.expon,
    }

    # récupérer params déjà fit
    rows = fit_table[fit_table["complexity"] == complexity_value]
    for _, r in rows.iterrows():
        name = r["model"]
        params = r["params"]
        if params is None or name not in candidates:
            continue

        dist = candidates[name]
        # pour pdf, on doit s'assurer que xs est positif si support positif
        xs_plot = xs.copy()
        if name in ["lognorm", "gamma", "weibull", "expon"]:
            xs_plot = _safe_pos(xs_plot)

        try:
            ys = dist.pdf(xs_plot, *params)
            ax.plot(xs_plot, ys, linewidth=2, label=name)
        except Exception:
            pass

    ax.set_title(f"{complexity_value} — histogramme + modèles ajustés")
    ax.set_xlabel(var)
    ax.set_ylabel("Densité")
    ax.legend()
    fig.tight_layout()
    return fig

from scipy import stats

def completion_probability(dist_name: str, params, target: float):
    """Retourne P(X <= target) pour une loi et ses paramètres scipy."""
    dist_map = {
        "normal": stats.norm,
        "lognorm": stats.lognorm,
        "gamma": stats.gamma,
        "weibull": stats.weibull_min,
        "expon": stats.expon,
    }
    if dist_name not in dist_map or params is None:
        return None

    dist = dist_map[dist_name]
    try:
        # support positif pour certaines lois
        if dist_name in ["lognorm", "gamma", "weibull", "expon"]:
            target = max(target, 1e-6)
        p = dist.cdf(target, *params)
        return float(p)
    except Exception:
        return None


def get_best_model_for_complexity(best_table: pd.DataFrame, complexity_value: str):
    """Récupère (model, params) depuis best_table pour une complexité donnée."""
    row = best_table[best_table["complexity"] == complexity_value]
    if row.empty:
        return None, None
    model = row.iloc[0]["model"]
    params = row.iloc[0]["params"]
    return model, params


def plot_cdf_comparison(best_model_name: str, best_params, target: float):
    """Petit graphe CDF: normal vs best model, avec un repère sur target."""
    dist_map = {
        "normal": stats.norm,
        "lognorm": stats.lognorm,
        "gamma": stats.gamma,
        "weibull": stats.weibull_min,
        "expon": stats.expon,
    }

    fig, ax = plt.subplots(figsize=(9, 4))

    # plage de x
    xmax = max(target * 2, target + 10)
    xs = np.linspace(0, xmax, 300)

    # Normal (standardisé autour de target pour affichage -> on ne peut pas sans params ici)
    # Donc on ne trace normal que si on a aussi ses params (à fournir depuis l'app).
    # Ici on trace uniquement le modèle optimal.
    if best_model_name in dist_map and best_params is not None:
        dist = dist_map[best_model_name]
        xs_plot = xs.copy()
        if best_model_name in ["lognorm", "gamma", "weibull", "expon"]:
            xs_plot = np.where(xs_plot <= 0, 1e-6, xs_plot)
        ys = dist.cdf(xs_plot, *best_params)
        ax.plot(xs_plot, ys, linewidth=2, label=f"CDF {best_model_name}")

        p = completion_probability(best_model_name, best_params, target)
        ax.axvline(target, linestyle="--")
        ax.scatter([target], [p], s=40)
        ax.set_title("Probabilité d'achèvement avant la cible (CDF)")
        ax.set_xlabel("Duration (mois)")
        ax.set_ylabel("P(Duration ≤ cible)")

        ax.legend()
        fig.tight_layout()
        return fig

    return fig

def evaluate_student_choice(fit_table: pd.DataFrame, complexity_value: str, chosen_model: str, criterion: str = "bic"):
    """
    Compare le choix étudiant au meilleur modèle (BIC par défaut).
    Retourne un dict avec le verdict + explication.
    """
    sub = fit_table[fit_table["complexity"] == complexity_value].copy()
    sub = sub.dropna(subset=[criterion])

    if sub.empty:
        return {"status": "Non testable", "message": "Pas assez de données pour comparer les modèles.", "delta": None}

    # meilleur modèle selon BIC (ou AIC)
    best_row = sub.loc[sub[criterion].idxmin()]
    best_model = best_row["model"]
    best_score = float(best_row[criterion])

    # score du choix étudiant
    stud_row = sub[sub["model"] == chosen_model]
    if stud_row.empty:
        return {"status": "Non testable", "message": "Modèle non trouvé dans les résultats.", "delta": None}

    stud_score = float(stud_row.iloc[0][criterion])
    delta = stud_score - best_score

    # règles simples et pédagogiques (BIC)
    if chosen_model == best_model:
        status = "✅ Correct"
        msg = f"Ton choix (**{chosen_model}**) est le meilleur selon {criterion.upper()}."
    else:
        # seuils usuels d'interprétation des écarts d'information criteria
        # delta < 2 : très proche ; 2-6 : modéré ; 6-10 : fort ; >10 : très fort
        if delta < 2:
            status = "🟡 Acceptable"
            msg = (
                f"Ton choix (**{chosen_model}**) n'est pas le meilleur, mais il est **très proche** du meilleur "
                f"selon {criterion.upper()} (Δ{criterion.upper()} = {delta:.2f})."
            )
        elif delta < 6:
            status = "🟠 Plutôt non optimal"
            msg = (
                f"Ton choix (**{chosen_model}**) est **moins bon** que le meilleur selon {criterion.upper()} "
                f"(Δ{criterion.upper()} = {delta:.2f})."
            )
        else:
            status = "❌ Non optimal"
            msg = (
                f"Ton choix (**{chosen_model}**) est **nettement moins adapté** que le meilleur selon {criterion.upper()} "
                f"(Δ{criterion.upper()} = {delta:.2f})."
            )

    # explication courte "pourquoi"
    extra = (
        f"PLUTO sélectionne **{best_model}** car il minimise {criterion.upper()} pour la complexité **{complexity_value}** "
        f"(meilleur {criterion.upper()} = {best_score:.2f} vs ton {criterion.upper()} = {stud_score:.2f})."
    )

    return {
        "status": status,
        "message": msg,
        "best_model": best_model,
        "best_score": best_score,
        "student_score": stud_score,
        "delta": delta,
        "extra": extra,
    }

@dataclass
class DurationModel:
    intercept: float
    coef_effort: float
    coef_cost: float
    coef_revisions: float

def fit_duration_linear_models(df: pd.DataFrame, group: str = "complexity") -> dict:
    """
    Ajuste un modèle linéaire simple:
      duration_months ≈ a + b1*effort_my + b2*cost_keur + b3*revisions_number
    par complexité. (Modèle "what-if", pédagogique)
    """
    required = ["duration_months", "effort_my", "cost_keur", "revisions_number", group]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour l'étape 5: {missing}")

    models = {}
    for comp, sub in df.groupby(group):
        sub = sub.dropna(subset=["duration_months", "effort_my", "cost_keur", "revisions_number"])
        if len(sub) < 8:
            continue

        y = sub["duration_months"].values.astype(float)
        X = np.column_stack([
            np.ones(len(sub)),
            sub["effort_my"].values.astype(float),
            sub["cost_keur"].values.astype(float),
            sub["revisions_number"].values.astype(float),
        ])

        # Moindres carrés (rapide, sans dépendance)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

        models[comp] = DurationModel(
            intercept=float(beta[0]),
            coef_effort=float(beta[1]),
            coef_cost=float(beta[2]),
            coef_revisions=float(beta[3]),
        )

    return models


def predict_duration(model: DurationModel, effort_my: float, cost_keur: float, revisions_number: float) -> float:
    """Prédiction durée (mois) via modèle linéaire."""
    pred = (
        model.intercept
        + model.coef_effort * effort_my
        + model.coef_cost * cost_keur
        + model.coef_revisions * revisions_number
    )
    return float(max(pred, 1.0))  # durée min 1 mois


def optimize_allocation(model: DurationModel,
                        effort_range: tuple,
                        cost_range: tuple,
                        rev_range: tuple,
                        step_effort: float = 0.5,
                        step_cost: float = 50.0,
                        step_rev: int = 1) -> dict:
    """
    Cherche la meilleure combinaison (effort, cost, revisions) qui minimise la durée prédite,
    par recherche grille (suffisant et lisible pour un outil formatif).
    """
    e_min, e_max = effort_range
    c_min, c_max = cost_range
    r_min, r_max = rev_range

    best = None
    for e in np.arange(e_min, e_max + 1e-9, step_effort):
        for c in np.arange(c_min, c_max + 1e-9, step_cost):
            for r in range(int(r_min), int(r_max) + 1, int(step_rev)):
                d = predict_duration(model, e, c, r)
                if (best is None) or (d < best["duration_pred"]):
                    best = {"effort_my": float(e), "cost_keur": float(c), "revisions_number": int(r), "duration_pred": float(d)}

    return best


def success_score(student_duration: float, optimal_duration: float) -> float:
    """
    Score 0–100 basé sur la proximité à l'optimum (durée).
    100 si égal à l'optimum, puis décroît avec l'écart relatif.
    """
    if optimal_duration <= 0:
        return 0.0
    err_rel = abs(student_duration - optimal_duration) / optimal_duration
    score = 100.0 * max(0.0, 1.0 - err_rel)
    return round(score, 1)
