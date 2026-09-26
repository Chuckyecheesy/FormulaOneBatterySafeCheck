"""Correlation analysis of the dataset against the XGBoost target (Efficiency (%)).

Run with:  uv run python -m formulatech.ml.correlation

Writes models/correlation_heatmap.png, models/correlation_with_target.png,
models/correlation_with_target.csv, models/random_forest_importance.png and
models/random_forest_importance.csv. Pearson and Spearman measure linear and monotonic
association; mutual information also catches non-linear signal that XGBoost can use.
Correlation is a property of the data, so it is the same for every model; the Random
Forest section instead measures how much a fitted model relies on each input.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.feature_selection import mutual_info_regression  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from formulatech.config import DATA_PATH, MODELS_DIR  # noqa: E402
from formulatech.ml.train import FEATURES, LEAKY_COLUMNS, RANDOM_STATE, TARGET, TEST_SIZE  # noqa: E402

# Diverging blue <-> red with a neutral gray midpoint (dataviz reference palette).
SURFACE = "#fcfcfb"
INK, INK_MUTED = "#0b0b0b", "#52514e"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#9c2222"]
)
BLUE, RED = "#2a78d6", "#e34948"


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson, Spearman and mutual information of every column against TARGET.

    Text columns are one-hot encoded so they can be scored too.
    """
    encoded = pd.concat(
        [df.select_dtypes("number"), pd.get_dummies(df.select_dtypes(exclude="number"), dtype=float)],
        axis=1,
    )
    X, y = encoded.drop(columns=[TARGET]), encoded[TARGET]
    table = pd.DataFrame({
        "pearson": X.corrwith(y),
        "spearman": X.corrwith(y, method="spearman"),
        "mutual_info": mutual_info_regression(X, y, random_state=RANDOM_STATE),
    })
    table["used_by_model"] = table.index.isin(FEATURES)
    table["leaky"] = table.index.isin(LEAKY_COLUMNS)
    return table.sort_values("pearson", key=np.abs, ascending=False)


def plot_heatmap(df: pd.DataFrame, path) -> None:
    """Lower-triangle Pearson matrix of the numeric columns, target last."""
    num = df.select_dtypes("number")
    num = num[[c for c in num.columns if c != TARGET] + [TARGET]]
    corr = num.corr()
    n = len(corr)
    masked = np.ma.masked_where(np.triu(np.ones((n, n), bool), k=1), corr.values)

    fig, ax = plt.subplots(figsize=(9.5, 8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(masked, cmap=DIVERGING, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(i + 1):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else INK)
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(n), corr.columns, rotation=40, ha="right", fontsize=9, color=INK_MUTED)
    ax.set_yticks(range(n), corr.columns, fontsize=9, color=INK_MUTED)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.get_yticklabels()[-1].set_color(INK)
    ax.get_yticklabels()[-1].set_fontweight("bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Pearson correlation", color=INK_MUTED)
    cbar.outline.set_visible(False)
    ax.set_title("Correlation between dataset columns (bottom row: efficiency target)",
                 color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_target_bars(table: pd.DataFrame, path) -> None:
    """Horizontal diverging bars: Pearson correlation of each column with the target."""
    t = table.sort_values("pearson", key=np.abs)
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    colors = [RED if v > 0 else BLUE for v in t["pearson"]]
    ax.barh(range(len(t)), t["pearson"], color=colors, height=0.7, edgecolor=SURFACE, linewidth=2)
    ax.axvline(0, color=INK_MUTED, linewidth=1)

    labels = []
    for name, row in t.iterrows():
        tag = " (leaks target)" if row["leaky"] else " (model input)" if row["used_by_model"] else ""
        labels.append(name + tag)
    ax.set_yticks(range(len(t)), labels, fontsize=9, color=INK_MUTED)
    for lbl, (_, row) in zip(ax.get_yticklabels(), t.iterrows()):
        if row["used_by_model"]:
            lbl.set_color(INK)
            lbl.set_fontweight("bold")
    for i, v in enumerate(t["pearson"]):  # direct labels only where the bar is meaningful
        if abs(v) >= 0.1:
            ax.text(v + (0.02 if v > 0 else -0.02), i, f"{v:+.2f}", va="center",
                    ha="left" if v > 0 else "right", fontsize=8, color=INK)

    ax.set_xlim(-1.15, 1.15)
    ax.set_xlabel("Pearson correlation with Efficiency (%)   (blue = negative, red = positive)",
                  color=INK_MUTED)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Which inputs relate to efficiency?", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def random_forest_importance(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Fit a Random Forest on every non-leaky column and score each input on the test split.

    Permutation importance = drop in test R² when that column is shuffled, so it is
    measured on unseen rows and is not biased towards high-cardinality columns the way
    impurity importance is.
    """
    X = pd.get_dummies(df.drop(columns=[TARGET, *LEAKY_COLUMNS]), dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, df[TARGET], test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    rf = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    perm = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
    table = pd.DataFrame(
        {
            "permutation_r2_drop": perm.importances_mean,
            "permutation_std": perm.importances_std,
            "impurity_importance": rf.feature_importances_,
            "used_by_model": X.columns.isin(FEATURES),
        },
        index=X.columns,
    ).sort_values("permutation_r2_drop", ascending=False)
    return table, r2_score(y_test, rf.predict(X_test))


def plot_rf_importance(table: pd.DataFrame, test_r2: float, path) -> None:
    """Horizontal bars (one hue: magnitude) of permutation importance, with ±1 std whiskers."""
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(range(len(t)), t["permutation_r2_drop"].clip(lower=0), xerr=t["permutation_std"],
            color=BLUE, height=0.7, edgecolor=SURFACE, linewidth=2,
            error_kw={"ecolor": INK_MUTED, "elinewidth": 1, "capsize": 2})
    labels = [name + (" (model input)" if used else "") for name, used in t["used_by_model"].items()]
    ax.set_yticks(range(len(t)), labels, fontsize=9, color=INK_MUTED)
    for lbl, used in zip(ax.get_yticklabels(), t["used_by_model"]):
        if used:
            lbl.set_color(INK)
            lbl.set_fontweight("bold")
    top = t["permutation_r2_drop"].max()
    for i, (v, sd) in enumerate(zip(t["permutation_r2_drop"], t["permutation_std"])):
        if v >= 0.005:  # direct labels only on visible bars, placed past the whisker
            ax.text(v + sd + top * 0.02, i, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, top * 1.2)
    ax.set_xlabel("Drop in test R² when the column is shuffled (permutation importance)", color=INK_MUTED)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Random Forest: which inputs does it rely on?\n"
                 f"Test R² = {test_r2:.3f}; {', '.join(LEAKY_COLUMNS)} excluded as leaky",
                 color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    table = correlation_table(df)
    table.round(4).to_csv(MODELS_DIR / "correlation_with_target.csv")
    plot_heatmap(df, MODELS_DIR / "correlation_heatmap.png")
    plot_target_bars(table, MODELS_DIR / "correlation_with_target.png")
    print(table.round(3).to_string())

    rf_table, rf_r2 = random_forest_importance(df)
    rf_table.round(4).to_csv(MODELS_DIR / "random_forest_importance.csv")
    plot_rf_importance(rf_table, rf_r2, MODELS_DIR / "random_forest_importance.png")
    print(f"\nRandom Forest test R² = {rf_r2:.4f}")
    print(rf_table.round(4).to_string())


if __name__ == "__main__":
    main()
