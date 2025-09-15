import numpy as np
import json
from tkinter import filedialog, Tk

def step_activation(u):
    """Función de activación escalón."""
    return 1 if u >= 0 else 0

def manual_simulation_file():
    print("=== Simulación manual con archivo de entrenamiento ===")

    # 1. Seleccionar archivo JSON
    Tk().withdraw()  # ocultar ventana raíz de Tk
    file_path = filedialog.askopenfilename(
        title="Seleccione el archivo JSON del entrenamiento",
        filetypes=[("JSON files", "*.json")]
    )
    if not file_path:
        print("No se seleccionó archivo. Saliendo...")
        return

    # 2. Leer archivo y obtener última iteración
    with open(file_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    last_iter = history[-1]  # última iteración
    weights = np.array(last_iter["weights"]).flatten()
    theta = np.array(last_iter["theta"]).flatten()

    print("\nÚltima iteración cargada:")
    print(f"Iteración: {last_iter['iter']}")
    print(f"RMS: {last_iter['rms']:.4f}")
    print(f"Pesos: {weights}")
    print(f"Umbrales θ: {theta}")

    n_inputs = len(weights)

    # 3. Ingresar patrón de prueba
    print("\nIngrese un patrón de prueba:")
    x = []
    for i in range(n_inputs):
        val = float(input(f"x{i+1}: "))
        x.append(val)
    x = np.array(x)

    # 4. Calcular salida
    u = np.dot(x, weights) - theta
    y = step_activation(u)

    print("\n--- Resultado ---")
    print(f"Entrada: {x}")
    print(f"Potencial neto u = {u}")
    print(f"Salida de la red (y) = {y}")

if __name__ == "__main__":
    manual_simulation_file()
