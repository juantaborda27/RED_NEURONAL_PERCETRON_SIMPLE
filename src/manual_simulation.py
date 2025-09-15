import numpy as np

def step_activation(u):
    """Función de activación escalón."""
    return 1 if u >= 0 else 0

def manual_simulation():
    print("=== Simulación manual del perceptrón ===")

    # 1. Definir pesos
    n_inputs = int(input("¿Cuántos pesos/entradas tendrá el perceptrón?: "))
    weights = []
    for i in range(n_inputs):
        val = float(input(f"Ingrese peso w{i+1}: "))
        weights.append(val)
    weights = np.array(weights)

    # 2. Definir umbral
    theta = float(input("Ingrese el umbral θ: "))

    print("\nConfiguración lista:")
    print(f"Pesos: {weights}")
    print(f"Umbral θ: {theta}")

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
    print(f"Potencial neto u = {u:.4f}")
    print(f"Salida de la red (y) = {y}")

if __name__ == "__main__":
    manual_simulation()
