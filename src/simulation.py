import numpy as np
import pandas as pd
from pathlib import Path
from data_utils import load_simple, dataset_summary

def step_activation(u):
    """Función de activación escalón (limitador duro)."""
    return 1 if u >= 0 else 0

def simulate_perceptron(dataset_path: str):
    # 1. Cargar dataset
    df = load_simple(Path(dataset_path))
    summary = dataset_summary(df)

    print("\n--- Resumen del dataset ---")
    print(f"Patrones: {summary['patrones']}")
    print(f"Entradas: {summary['entradas']}")
    print(f"Salidas: {summary['salidas']}")
    print(f"Columnas: {summary['columns']}\n")

    input_cols = summary["input_cols"]
    target_col = summary["target_col"]

    X = df[input_cols].to_numpy(dtype=float)  # (N, n_inputs)
    D = df[[target_col]].to_numpy(dtype=float)  # (N, 1)

    n_inputs = summary["entradas"]
    n_outputs = summary["salidas"]

    # 2. Solicitar pesos y umbrales
    print("Ingrese los pesos entrenados:")
    weights = np.zeros((n_inputs, n_outputs))
    for j in range(n_outputs):
        for i in range(n_inputs):
            val = float(input(f"w{i+1}{j+1} (peso de entrada {i+1} hacia salida {j+1}): "))
            weights[i, j] = val

    print("\nIngrese los umbrales entrenados:")
    theta = np.zeros(n_outputs)
    for j in range(n_outputs):
        theta[j] = float(input(f"θ{j+1}: "))

    # 3. Calcular salida de la red
    print("\n--- Resultados de la simulación ---")
    Y = []
    for idx, x in enumerate(X):
        outputs = []
        for j in range(n_outputs):
            u = np.dot(x, weights[:, j]) - theta[j]
            y = step_activation(u)
            outputs.append(y)
        Y.append(outputs)
        print(f"Patrón {idx+1}: Entrada={x}, Deseada={D[idx]}, Red={outputs}")

    Y = np.array(Y)

    # 4. Métricas simples (porcentaje de aciertos)
    aciertos = np.sum(Y.flatten() == D.flatten())
    total = len(D.flatten())
    print(f"\nPrecisión: {aciertos}/{total} ({(aciertos/total)*100:.2f}%)")

    return Y

if __name__ == "__main__":
    path = input("Ingrese la ruta del dataset a simular: ")
    simulate_perceptron(path)
