# ===============================
# CASO B: PREDICCIÓN DE CHURN
# ===============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Semilla para replicabilidad
SEED = 42

# Cargar el archivo de entrenamiento
df = pd.read_csv("train.csv")

# Ver los primeros datos y la información general
print(df.head())
print(df.info())
print(df.isnull().sum())

# Convertir TotalCharges a numérico
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Eliminar filas con valores nulos
df = df.dropna()

# Eliminar customerID porque no aporta al modelo
df = df.drop("customerID", axis=1)

# Ver las columnas disponibles
print(df.columns.tolist())

# Convertir Churn a binario (1 = Yes, 0 = No)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Definir X (predictoras) e y (variable objetivo)
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Separar las variables numéricas y categóricas
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Mostrar las variables
print("Variables numéricas:", numeric_features)
print("Variables categóricas:", categorical_features)

# Preprocesamiento de datos
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# División train/test (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Modelos a probar
models = {
    "Regresión Logística": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=SEED
    ),
    "Árbol de Decisión": DecisionTreeClassifier(
        class_weight="balanced", random_state=SEED
    )
}

results = []

# Entrenamiento y evaluación de modelos
for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        "Modelo": name,
        "Accuracy": acc,
        "F1-score": f1
    })

    # Mostrar resultados para cada modelo
    print("\n====================")
    print(name)
    print("====================")
    print("Accuracy:", acc)
    print("F1-score:", f1)
    print("Matriz de confusión:")
    print(cm)
    print(classification_report(y_test, y_pred))

# Mostrar la tabla final con los resultados
results_df = pd.DataFrame(results)
print(results_df)

# Gráfico de la matriz de confusión para el último modelo
plt.figure(figsize=(8, 6))
plt.title(f"Matriz de Confusión: {name}")
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(np.arange(2), ['No', 'Sí'])
plt.yticks(np.arange(2), ['No', 'Sí'])
plt.ylabel('Actual')
plt.xlabel('Predicho')

# Añadir etiquetas de las celdas de la matriz de confusión
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="white")

plt.show()