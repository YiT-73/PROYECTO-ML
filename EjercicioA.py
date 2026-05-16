import numpy as np

#Parte 1 Regresión Lineal Múltiple
print("\nRegresión Lineal Múltiple")
X_raw = np.array([
    [95,  4, 13, 12.5],
    [123, 2, 14,  8.0],
    [118, 2, 22, 15.0],
    [96,  4,  8,  5.5],
    [182, 4, 19,  9.0],
    [86,  4,  5,  3.5],
    [63,  2,  7, 18.0],
    [193, 2,  3,  4.0],
    [155, 2, 29, 22.0],
    [128, 2,  3,  6.5],
    [195, 5, 18,  7.5],
    [115, 5, 22, 20.0],
    [105, 3, 10,  2.0],
    [140, 3,  6, 14.0]
], dtype=float)

y = np.array([
    171600, 216200, 196900, 190200,
    305700, 180100, 133600, 320000,
    242700, 216800, 357400, 219800,
    198500, 241300
], dtype=float)

# Normalización Z-score de las variables independientes
media = X_raw.mean(axis=0)
desviacion = X_raw.std(axis=0)

X_norm = (X_raw - media) / desviacion

# Agregamos columna de 1 para w0
X = np.c_[np.ones(X_norm.shape[0]), X_norm]

#------------Ecuación normal-----------
print("\nEcuación normal")
# Transpuesta
XT = X.T

# Producto X^T X
XTX = XT @ X

# Producto X^T y
XTy = XT @ y

# Cálculo de pesos por ecuación normal
# (X^T X)^-1 XT Y
w_normal = np.linalg.inv(XTX) @ XTy
print("\nPesos finales Ecuación normal:")
print(w_normal)

#------------Gradiente decent-----------

print("\nGradiente decent")

# Parámetros iniciales
m, n = X.shape
#w0=0
w_gd = np.zeros(n)
lr = 0.01
epochs = 5000
# Gradient Descent
for epoch in range(1, epochs + 1):

    # Predicción
    y_pred = X @ w_gd

    # Error
    error = y_pred - y

    # MSE
    mse = np.mean(error ** 2)

    # Gradiente
    gradiente = (2 / m) * (X.T @ error)

    # Actualización de pesos
    w_gd = w_gd - lr * gradiente

    # Mostrar evolución cada 500 épocas
    if epoch % 500 == 0:
        print(f"Época {epoch}: MSE = {mse:.2f}")

print("\nPesos finales GD:")
print(w_gd)


#Predicción
print("\nPredcción:")
x_new = np.array([[120, 3, 10, 7]], dtype=float)
print("\nCaso:")
print(x_new)
# Normalizar usando la misma media y desviación del entrenamiento
x_new_norm = (x_new - media) / desviacion

# Agregar columna de 1 para w0
x_new_final = np.c_[np.ones(x_new_norm.shape[0]), x_new_norm]

# Predicción
precio_predicho = x_new_final @ w_normal
print("\nPredcción final:")
print(precio_predicho)


#Parte 2 Regresión Logística
print("\nRegresión Logística")
# Variable objetivo binaria
y_bin = (y > 200000).astype(int).reshape(-1, 1)
print("\nY binario:")
print(y_bin.ravel())
clase_0 = np.sum(y_bin == 0)
clase_1 = np.sum(y_bin == 1)

print("Clase 0 - ESTÁNDAR:", clase_0)
print("Clase 1 - PREMIUM:", clase_1)


print("\nModelos en solitario:")

# =========================
# Funciones
# =========================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y_true, p):
    eps = 1e-15
    return -np.mean(
        y_true * np.log(p + eps) +
        (1 - y_true) * np.log(1 - p + eps)
    )

def modelo_logistico_univariado(x, y_bin, lr=0.01, epochs=2000):
    # Asegurar vectores columna
    x = x.reshape(-1, 1)
    y_bin = y_bin.reshape(-1, 1)

    # Normalización Z-score
    media = np.mean(x)
    desviacion = np.std(x)
    x_norm = (x - media) / desviacion

    # Matriz con intercepto
    X = np.c_[np.ones((x_norm.shape[0], 1)), x_norm]

    # Inicializar pesos
    w = np.zeros((2, 1))

    m = X.shape[0]

    for epoch in range(epochs):
        z = X @ w
        p = sigmoid(z)

        gradiente = (1 / m) * (X.T @ (p - y_bin))

        w = w - lr * gradiente

    # Predicción final
    p_final = sigmoid(X @ w)
    y_pred = (p_final >= 0.5).astype(int)

    accuracy = np.mean(y_pred == y_bin)
    bce = binary_cross_entropy(y_bin, p_final)

    return w, accuracy, bce


def entrenar_logistica_completa(X_raw, y_bin, lr=0.01, epochs=3000):
    # Normalización Z-score
    media = X_raw.mean(axis=0)
    desviacion = X_raw.std(axis=0)

    X_norm = (X_raw - media) / desviacion

    # Agregar columna de 1 para w0
    X = np.c_[np.ones((X_norm.shape[0], 1)), X_norm]

    # Inicializar pesos
    w = np.zeros((X.shape[1], 1))

    m = X.shape[0]

    for epoch in range(epochs):
        z = X @ w
        p = sigmoid(z)

        gradiente = (1 / m) * (X.T @ (p - y_bin))

        w = w - lr * gradiente

    # Predicciones finales
    p_final = sigmoid(X @ w)
    y_pred = (p_final >= 0.5).astype(int)

    accuracy = np.mean(y_pred == y_bin)

    return w, accuracy, y_pred, p_final, media, desviacion



# =========================
# Entrenar modelos
# =========================

nombres = ["Área", "Habitaciones", "Antigüedad", "Distancia"]


for i, nombre in enumerate(nombres):
    x = X_raw[:, i]

    w, accuracy, bce = modelo_logistico_univariado(
        x, y_bin, lr=0.01, epochs=2000
    )

    print(f"\nModelo con solo {nombre}")
    print(f"w0 = {w[0, 0]:.4f}")
    print(f"w{i+1} = {w[1, 0]:.4f}")
    print(f"Accuracy = {accuracy:.4f}")
    print(f"BCE = {bce:.4f}")


# =========================
# Modelo completo
# =========================
print("\nModelo completo:")
w_full, acc_full, y_pred_full, p_full, media, desviacion = entrenar_logistica_completa(
    X_raw, y_bin, lr=0.01, epochs=3000
)

print("Pesos del modelo completo:")
print(f"w0 = {w_full[0,0]:.4f}")
print(f"w1 = {w_full[1,0]:.4f}")
print(f"w2 = {w_full[2,0]:.4f}")
print(f"w3 = {w_full[3,0]:.4f}")
print(f"w4 = {w_full[4,0]:.4f}")

print("\nAccuracy modelo completo:")
print(f"{acc_full:.4f}")

# =========================
# Matriz de confusión 2x2
# Filas: clase real
# Columnas: clase predicha
# =========================

TN = np.sum((y_bin == 0) & (y_pred_full == 0))
FP = np.sum((y_bin == 0) & (y_pred_full == 1))
FN = np.sum((y_bin == 1) & (y_pred_full == 0))
TP = np.sum((y_bin == 1) & (y_pred_full == 1))

matriz_confusion = np.array([
    [TN, FP],
    [FN, TP]
])

print("Matriz de confusión:")
print(matriz_confusion)

# =========================
# Predicción propiedad nueva
# =========================

x_nueva = np.array([[175, 4, 8, 6]], dtype=float)

# Normalizar con la misma media y desviación del entrenamiento
x_nueva_norm = (x_nueva - media) / desviacion

# Agregar columna de 1
x_nueva_final = np.c_[np.ones((1, 1)), x_nueva_norm]

# Probabilidad PREMIUM
p_premium = sigmoid(x_nueva_final @ w_full)

clase = (p_premium >= 0.5).astype(int)

print(f"P(PREMIUM) = {p_premium[0,0]:.4f}")
print(f"Clase predicha = {clase[0,0]}")
