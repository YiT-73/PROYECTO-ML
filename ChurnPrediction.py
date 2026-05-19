import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, f1_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

SEED = 42
np.random.seed(SEED)

print("=" * 60)
print("1. CARGA DE DATOS")
print("=" * 60)

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

print(f"  Train : {train.shape}")
print(f"  Test  : {test.shape}")

#LIMPIEZA DE DATOS
print("\n" + "=" * 60)
print("2. LIMPIEZA DE DATOS")
print("=" * 60)

def clean_data(df, is_train=True):
    df = df.copy()

    # Eliminar ID: no aporta poder predictivo
    df.drop(columns=["customerID"], inplace=True)

    # TotalCharges: almacenada como string con espacios donde tenure=0.
    # Imputamos con MonthlyCharges × 1 (primer mes del cliente nuevo).
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].str.strip(), errors="coerce"
    )
    mask_tc = df["TotalCharges"].isna()
    df.loc[mask_tc, "TotalCharges"] = df.loc[mask_tc, "MonthlyCharges"] * 1
    print(f"  TotalCharges imputados : {mask_tc.sum()} fila(s)")

    # Codificar target solo en train
    if is_train:
        df["Churn"] = (df["Churn"].str.strip() == "Yes").astype(int)

    return df

train_clean = clean_data(train, is_train=True)
test_clean  = clean_data(test,  is_train=False)

TARGET = "Churn"
X = train_clean.drop(columns=[TARGET])
y = train_clean[TARGET]

vc = y.value_counts()
print(f"\n  Distribución de clases:")
print(f"    No Churn (0) : {vc[0]:>5}  ({vc[0]/len(y)*100:.1f}%)")
print(f"    Churn    (1) : {vc[1]:>5}  ({vc[1]/len(y)*100:.1f}%)")
print(f"  Ratio desbalance : {vc[0]/vc[1]:.2f}:1")

# Definicion de columnas por tipo
binary_cols = [
    "gender", "Partner", "Dependents",
    "PhoneService", "PaperlessBilling"
]
ordinal_cols = [
    "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod"
]
numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

feature_names = numeric_cols + binary_cols + ordinal_cols

print(f"\n  Features numéricas  : {len(numeric_cols)}")
print(f"  Features binarias   : {len(binary_cols)}")
print(f"  Features ordinales  : {len(ordinal_cols)}")

#PRe procesador
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(),
         numeric_cols),
        ("bin", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1),
         binary_cols),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1),
         ordinal_cols),
    ],
    remainder="drop"
)

# Se generan muestras sintéticas de la clase minoritaria.
smote = SMOTE(sampling_strategy=0.6, random_state=SEED, k_neighbors=5)

#Modelos a usar

# --- Random Forest ---
rf_model = RandomForestClassifier(
    n_estimators      = 500,
    max_depth         = None,
    min_samples_split = 10,
    min_samples_leaf  = 4,
    max_features      = "sqrt",
    class_weight      = "balanced",
    random_state      = SEED,
    n_jobs            = -1
)

# --- Regresión Logística ---
lr_model = LogisticRegression(
    C            = 1.0,
    max_iter     = 1000,
    class_weight = "balanced",
    solver       = "lbfgs",
    random_state = SEED,
    n_jobs       = -1
)

# --- Pipelines con SMOTE ---
pipeline_rf = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote",        smote),
    ("model",        rf_model)
])

pipeline_lr = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote",        smote),
    ("model",        lr_model)
])

print("\n" + "=" * 60)
print("3. CROSS-VALIDATION ESTRATIFICADA (StratifiedKFold, k=5)")
print("=" * 60)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print("\n  Calculando predicciones OOF — Random Forest...")
oof_probs_rf = cross_val_predict(
    pipeline_rf, X, y, cv=CV, method="predict_proba", n_jobs=-1
)[:, 1]

print("  Calculando predicciones OOF — Regresión Logística...")
oof_probs_lr = cross_val_predict(
    pipeline_lr, X, y, cv=CV, method="predict_proba", n_jobs=-1
)[:, 1]

auc_rf = roc_auc_score(y, oof_probs_rf)
auc_lr = roc_auc_score(y, oof_probs_lr)
apr_rf = average_precision_score(y, oof_probs_rf)
apr_lr = average_precision_score(y, oof_probs_lr)
f1_rf_default = f1_score(y, (oof_probs_rf >= 0.5).astype(int))
f1_lr_default = f1_score(y, (oof_probs_lr >= 0.5).astype(int))

print(f"\n  {'Métrica':<30} {'Random Forest':>15} {'Log. Regresión':>15}")
print(f"  {'-'*60}")
print(f"  {'ROC-AUC (OOF)':<30} {auc_rf:>15.4f} {auc_lr:>15.4f}")
print(f"  {'Avg Precision (OOF)':<30} {apr_rf:>15.4f} {apr_lr:>15.4f}")
print(f"  {'F1 Churn (th=0.5)':<30} {f1_rf_default:>15.4f} {f1_lr_default:>15.4f}")

print("\n" + "=" * 60)
print("4. OPTIMIZACIÓN DE THRESHOLD (Matriz de Costo)")
print("=" * 60)

C_FN = 5   # costo de no detectar un churn real
C_FP = 1   # costo de ofrecer incentivo innecesario

thresholds = np.linspace(0.05, 0.90, 500)

def optimizar_threshold(probs, y_true, c_fn, c_fp):
    costs = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        costs.append(c_fn * fn + c_fp * fp)
    costs = np.array(costs)
    best_idx = np.argmin(costs)
    return thresholds[best_idx], costs, costs[best_idx]

best_th_rf, costs_rf, min_cost_rf = optimizar_threshold(
    oof_probs_rf, y, C_FN, C_FP
)
best_th_lr, costs_lr, min_cost_lr = optimizar_threshold(
    oof_probs_lr, y, C_FN, C_FP
)

print(f"\n  Costo FN (churn no detectado) : {C_FN}x")
print(f"  Costo FP (falsa alarma)       : {C_FP}x")
print(f"\n  {'Modelo':<25} {'Threshold óptimo':>18} {'Costo mínimo':>14}")
print(f"  {'-'*58}")
print(f"  {'Random Forest':<25} {best_th_rf:>18.4f} {min_cost_rf:>14,.0f}")
print(f"  {'Regresión Logística':<25} {best_th_lr:>18.4f} {min_cost_lr:>14,.0f}")

oof_preds_rf = (oof_probs_rf >= best_th_rf).astype(int)
oof_preds_lr = (oof_probs_lr >= best_th_lr).astype(int)

f1_rf_opt = f1_score(y, oof_preds_rf)
f1_lr_opt = f1_score(y, oof_preds_lr)

print(f"\n  F1 Churn con threshold óptimo:")
print(f"    Random Forest       : {f1_rf_opt:.4f}  (vs {f1_rf_default:.4f} con 0.5)")
print(f"    Regresión Logística : {f1_lr_opt:.4f}  (vs {f1_lr_default:.4f} con 0.5)")
    
print("\n" + "=" * 60)
print("5. REPORTE DE CLASIFICACIÓN (OOF, threshold óptimo)")
print("=" * 60)

cm_rf = confusion_matrix(y, oof_preds_rf)
cm_lr = confusion_matrix(y, oof_preds_lr)

tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()
tn_lr, fp_lr, fn_lr, tp_lr = cm_lr.ravel()

recall_rf    = tp_rf / (tp_rf + fn_rf)
precision_rf = tp_rf / (tp_rf + fp_rf)
recall_lr    = tp_lr / (tp_lr + fn_lr)
precision_lr = tp_lr / (tp_lr + fp_lr)

print("\n  --- Random Forest ---")
print(classification_report(y, oof_preds_rf,
                             target_names=["No Churn", "Churn"]))
print("  --- Regresión Logística ---")
print(classification_report(y, oof_preds_lr,
                             target_names=["No Churn", "Churn"]))


# ---------------------------------------------------------------------
print("=" * 60)
print("6. ENTRENAMIENTO FINAL Y PREDICCIONES EN TEST")
print("=" * 60)

pipeline_rf.fit(X, y)
pipeline_lr.fit(X, y)

X_test = test_clean.drop(
    columns=[c for c in ["Churn", "customerID"] if c in test_clean.columns]
)

test_probs_rf = pipeline_rf.predict_proba(X_test)[:, 1]
test_probs_lr = pipeline_lr.predict_proba(X_test)[:, 1]
test_preds_rf = (test_probs_rf >= best_th_rf).astype(int)
test_preds_lr = (test_probs_lr >= best_th_lr).astype(int)

print(f"\n  RF — Churn predichos: {test_preds_rf.sum()} / {len(test_preds_rf)}"
      f"  ({test_preds_rf.mean()*100:.1f}%)")
print(f"  LR — Churn predichos: {test_preds_lr.sum()} / {len(test_preds_lr)}"
      f"  ({test_preds_lr.mean()*100:.1f}%)")

results = pd.DataFrame({
    "customerID" : test["customerID"],
    "Prob_RF"    : np.round(test_probs_rf, 4),
    "Pred_RF"    : ["Yes" if p == 1 else "No" for p in test_preds_rf],
    "Prob_LR"    : np.round(test_probs_lr, 4),
    "Pred_LR"    : ["Yes" if p == 1 else "No" for p in test_preds_lr],
})
results.to_csv("predicciones_churn.csv", index=False)
print("\n  ✓ predicciones_churn.csv guardado")

# Feature Importance del Random Forest
importances = pipeline_rf.named_steps["model"].feature_importances_
feat_df = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

# Graficos
print("\n" + "=" * 60)
print("7. GENERANDO GRÁFICOS")
print("=" * 60)

C_MAIN   = "#2563EB"
C_ACCENT = "#EF4444"
C_GREEN  = "#16A34A"
C_ORANGE = "#F97316"
C_GRAY   = "#94A3B8"
C_BG     = "#F8FAFC"

plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "font.size"       : 11,
    "axes.titlesize"  : 13,
    "axes.labelsize"  : 11,
    "xtick.labelsize" : 10,
    "ytick.labelsize" : 10,
    "legend.fontsize" : 10,
    "figure.facecolor": C_BG,
    "axes.facecolor"  : "white",
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
    "grid.linestyle"  : "--",
})

# Heatmap y matriz de confusion
fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5.5))
fig1.patch.set_facecolor(C_BG)
fig1.suptitle(
    "Matrices de Confusión — OOF con Threshold Óptimo por Matriz de Costo",
    fontsize=14, fontweight="bold", y=1.01
)

def plot_cm(ax, cm, title, threshold, recall, precision, auc):
    cm_pct = cm.astype(float) / cm.sum()
    labels = np.array([
        [f"{cm[i,j]:,}\n({cm_pct[i,j]*100:.1f}%)" for j in range(2)]
        for i in range(2)
    ])
    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues",
        linewidths=1.5, linecolor="#CBD5E1",
        cbar_kws={"shrink": 0.8, "label": "N° muestras"},
        ax=ax, annot_kws={"size": 12, "weight": "bold"}
    )
    ax.set_title(f"{title}\n(Threshold = {threshold:.2f})",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Clase Predicha", fontsize=11, labelpad=6)
    ax.set_ylabel("Clase Real",     fontsize=11, labelpad=6)
    ax.set_xticklabels(["No Churn", "Churn"], fontsize=11)
    ax.set_yticklabels(["No Churn", "Churn"], fontsize=11, rotation=0)
    info = (f"Recall Churn: {recall:.2%}\n"
            f"Precisión:    {precision:.2%}\n"
            f"ROC-AUC:      {auc:.4f}")
    ax.text(
        2.55, 0.5, info, transform=ax.transData,
        fontsize=10, verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#DBEAFE",
                  edgecolor=C_MAIN, alpha=0.9)
    )

plot_cm(axes1[0], cm_rf, "Random Forest",
        best_th_rf, recall_rf, precision_rf, auc_rf)
plot_cm(axes1[1], cm_lr, "Regresión Logística",
        best_th_lr, recall_lr, precision_lr, auc_lr)

plt.tight_layout()
plt.savefig("grafico1_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ grafico1_confusion_matrix.png")

#Precission call y curvas
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5))
fig2.patch.set_facecolor(C_BG)
fig2.suptitle(
    "Evaluación del Modelo — Curva ROC y Curva Precision-Recall (OOF)",
    fontsize=14, fontweight="bold", y=1.01
)

ax_roc = axes2[0]
for probs, th, label, color in [
    (oof_probs_rf, best_th_rf,
     f"Random Forest (AUC={auc_rf:.4f})",  C_MAIN),
    (oof_probs_lr, best_th_lr,
     f"Log. Regresión (AUC={auc_lr:.4f})", C_ORANGE),
]:
    fpr, tpr, roc_t = roc_curve(y, probs)
    ax_roc.plot(fpr, tpr, lw=2.5, color=color, label=label)
    idx = np.argmin(np.abs(roc_t - th))
    ax_roc.scatter(fpr[idx], tpr[idx], s=100, zorder=5, color=color,
                   edgecolors="white", linewidths=1.5)

ax_roc.plot([0,1],[0,1], color=C_GRAY, lw=1.5, linestyle="--",
            label="Clasificador aleatorio (AUC=0.50)")
ax_roc.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=11)
ax_roc.set_ylabel("Tasa de Verdaderos Positivos (Recall)", fontsize=11)
ax_roc.set_title("Curva ROC", fontsize=13, fontweight="bold")
ax_roc.legend(fontsize=10, loc="lower right")
ax_roc.set_xlim([-0.02, 1.02])
ax_roc.set_ylim([-0.02, 1.05])

ax_pr = axes2[1]
for probs, th, apr, label, color in [
    (oof_probs_rf, best_th_rf, apr_rf,
     f"Random Forest (APR={apr_rf:.4f})",  C_MAIN),
    (oof_probs_lr, best_th_lr, apr_lr,
     f"Log. Regresión (APR={apr_lr:.4f})", C_ORANGE),
]:
    prec_c, rec_c, pr_t = precision_recall_curve(y, probs)
    ax_pr.plot(rec_c, prec_c, lw=2.5, color=color, label=label)
    idx = np.argmin(np.abs(pr_t - th))
    ax_pr.scatter(rec_c[idx], prec_c[idx], s=100, zorder=5, color=color,
                  edgecolors="white", linewidths=1.5)

baseline = y.mean()
ax_pr.axhline(baseline, color=C_GRAY, lw=1.5, linestyle="--",
              label=f"Baseline (prevalencia={baseline:.2f})")
ax_pr.set_xlabel("Recall (Sensibilidad)", fontsize=11)
ax_pr.set_ylabel("Precisión", fontsize=11)
ax_pr.set_title("Curva Precision-Recall", fontsize=13, fontweight="bold")
ax_pr.legend(fontsize=10, loc="upper right")
ax_pr.set_xlim([-0.02, 1.02])
ax_pr.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig("grafico2_roc_pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ grafico2_roc_pr_curves.png")


# Nuevo grafico
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
fig3.patch.set_facecolor(C_BG)
fig3.suptitle(
    "Importancia de Variables (RF) y Análisis de Costo por Threshold",
    fontsize=14, fontweight="bold", y=1.01
)

ax_fi = axes3[0]
top_n    = 12
top_feat = feat_df.head(top_n)
bar_colors = [C_MAIN if i < 5 else C_GRAY for i in range(top_n)]

bars = ax_fi.barh(
    top_feat["feature"][::-1],
    top_feat["importance"][::-1],
    color=bar_colors[::-1],
    edgecolor="white", linewidth=0.5, height=0.7
)
ax_fi.set_xlabel("Importancia (Gini impurity reduction)", fontsize=11)
ax_fi.set_title(f"Top {top_n} Variables — Random Forest",
                fontsize=12, fontweight="bold")
ax_fi.set_xlim([0, top_feat["importance"].max() * 1.18])

for bar in bars:
    w = bar.get_width()
    ax_fi.text(w + 0.0005, bar.get_y() + bar.get_height()/2,
               f"{w:.4f}", va="center", ha="left", fontsize=9.5)

from matplotlib.patches import Patch
legend_fi = [
    Patch(facecolor=C_MAIN, label="Top 5 variables"),
    Patch(facecolor=C_GRAY, label="Resto")
]
ax_fi.legend(handles=legend_fi, fontsize=10, loc="lower right")
ax_fi.grid(axis="x", alpha=0.3)
ax_fi.grid(axis="y", alpha=0)

ax_cost = axes3[1]
ax_cost.plot(thresholds, costs_rf, color=C_MAIN,   lw=2.5,
             label="Random Forest")
ax_cost.plot(thresholds, costs_lr, color=C_ORANGE, lw=2.5,
             linestyle="--", label="Regresión Logística")

for th, cost, color, name in [
    (best_th_rf, min_cost_rf, C_MAIN,   "RF"),
    (best_th_lr, min_cost_lr, C_ORANGE, "LR"),
]:
    ax_cost.axvline(th, color=color, lw=1.5, linestyle=":", alpha=0.7)
    ax_cost.scatter([th], [cost], s=100, zorder=5, color=color,
                    edgecolors="white", linewidths=1.5)
    ax_cost.annotate(
        f"{name}: th={th:.2f}\nCosto={cost:,.0f}",
        xy=(th, cost),
        xytext=(th + 0.05, cost + (costs_rf.max()-costs_rf.min())*0.08),
        fontsize=9.5,
        arrowprops=dict(arrowstyle="->", color="gray"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F1F5F9",
                  edgecolor=color, alpha=0.9)
    )

ax_cost.axvline(0.5, color=C_GRAY, lw=1.5, linestyle=":",
                label="Threshold por defecto (0.50)")
ax_cost.set_xlabel("Threshold de Decisión", fontsize=11)
ax_cost.set_ylabel(f"Costo Total (FN×{C_FN} + FP×{C_FP})", fontsize=11)
ax_cost.set_title(
    f"Curva de Costo vs Threshold\n(Matriz de Costo: C_FN={C_FN}, C_FP={C_FP})",
    fontsize=12, fontweight="bold"
)
ax_cost.legend(fontsize=10)

plt.tight_layout()
plt.savefig("grafico3_features_cost.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ grafico3_features_cost.png")

#Resumen final
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"\n  {'Métrica':<30} {'Random Forest':>15} {'Log. Regresión':>16}")
print(f"  {'-'*62}")
print(f"  {'ROC-AUC (OOF)':<30} {auc_rf:>15.4f} {auc_lr:>16.4f}")
print(f"  {'Avg Precision (OOF)':<30} {apr_rf:>15.4f} {apr_lr:>16.4f}")
print(f"  {'F1 Churn (th óptimo)':<30} {f1_rf_opt:>15.4f} {f1_lr_opt:>16.4f}")
print(f"  {'Recall Churn':<30} {recall_rf:>15.4f} {recall_lr:>16.4f}")
print(f"  {'Precisión Churn':<30} {precision_rf:>15.4f} {precision_lr:>16.4f}")
print(f"  {'Threshold óptimo':<30} {best_th_rf:>15.4f} {best_th_lr:>16.4f}")
print(f"  {'Costo mínimo':<30} {min_cost_rf:>15,.0f} {min_cost_lr:>16,.0f}")
print("\n  Script finalizado correctamente.")
