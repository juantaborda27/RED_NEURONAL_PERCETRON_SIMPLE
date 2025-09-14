# src/trainer.py
"""
Trainer para Perceptron unicapa usando la Regla Delta (Corrección de error).
Función principal: train_perceptron(X, d, w, theta, eta, max_iter, epsilon, callback=None, verbose=False)

- X: numpy array shape (N, n_inputs)  (SIN bias)
- d: numpy array shape (N,) o (N, n_outputs) con valores 0/1
- w: numpy array shape (n_inputs, n_outputs)
- theta: numpy array shape (n_outputs,)
- eta: learning rate (float)
- max_iter: máximo de épocas (int)
- epsilon: tolerancia RMS para detener (float)
- callback: función opcional con firma callback(iteration, rms, w, theta)
            llamada al final de cada iteración (época).
- verbose: imprime progresos por consola si True

Devuelve:
    history: dict { "rms": list_of_rms, "w": final_weights, "theta": final_theta, "iterations": n_iter_done }
"""

import numpy as np
from typing import Callable, Optional, Dict, Any
import math

def _ensure_2d_targets(d: np.ndarray) -> np.ndarray:
    """Convierte d a forma (N, n_outputs). Si d es (N,), vuelve a (N,1)."""
    d = np.asarray(d)
    if d.ndim == 1:
        return d.reshape((-1, 1))
    return d

def step_function(net: np.ndarray) -> np.ndarray:
    """Función escalón: devuelve 1 donde net >= 0, 0 donde net < 0"""
    return (net >= 0).astype(float)

def compute_rms(d: np.ndarray, y: np.ndarray) -> float:
    """
    RMS según la guía (error promedio por patrón).
    d and y shapes: (N, n_outputs)
    MSE = mean over all elements (d - y)^2
    RMS = sqrt(MSE)
    """
    diff = d - y
    mse = np.mean(np.square(diff))
    return math.sqrt(mse)

def train_perceptron(
    X: np.ndarray,
    d: np.ndarray,
    w: np.ndarray,
    theta: np.ndarray,
    eta: float = 0.1,
    max_iter: int = 1000,
    epsilon: float = 1e-3,
    callback: Optional[Callable[[int, float, np.ndarray, np.ndarray], None]] = None,
    verbose: bool = False,
    shuffle: bool = True
) -> Dict[str, Any]:
    """
    Entrena usando regla delta (perceptrón clásico por patrón).
    Actualizaciones (por patrón i):
      net = x_i @ w           -> shape (n_outputs,)
      y = step(net - theta)
      e = d_i - y             -> shape (n_outputs,)
      w += eta * outer(x_i, e)
      theta -= eta * e

    Al final de cada iteración (época) se calcula RMS sobre todo el dataset y se
    llama al callback con (iter, rms, w.copy(), theta.copy()) si callback != None.
    """
    # Validaciones y normalizaciones de forma
    X = np.asarray(X, dtype=float)
    N, n_inputs = X.shape
    d2 = _ensure_2d_targets(d)       # (N, n_outputs)
    n_outputs = d2.shape[1]

    if w.shape != (n_inputs, n_outputs):
        raise ValueError(f"weights w shape debe ser (n_inputs, n_outputs) = ({n_inputs},{n_outputs}), got {w.shape}")
    if theta.shape != (n_outputs,):
        raise ValueError(f"theta shape debe ser (n_outputs,) = ({n_outputs},), got {theta.shape}")

    # historial
    rms_history = []
    iter_done = 0

    # Entrenamiento (épocas)
    for epoch in range(1, max_iter + 1):
        iter_done = epoch
        # opcionalmente barajar los índices
        indices = np.arange(N)
        if shuffle:
            np.random.shuffle(indices)

        # actualizar por patrón (online / estocástico)
        for i in indices:
            x_i = X[i]                     # (n_inputs,)
            d_i = d2[i]                    # (n_outputs,)
            net = x_i @ w                  # (n_outputs,)
            # comparamos net con theta: y = step(net - theta)
            y_i = step_function(net - theta)  # (n_outputs,)
            e = d_i - y_i                  # (n_outputs,)
            # actualizar pesos: w += eta * outer(x_i, e)
            # outer -> (n_inputs, n_outputs)
            w += eta * np.outer(x_i, e)
            # actualizar umbral: theta -= eta * e
            theta -= eta * e

        # al final de la época, medir RMS sobre todo el dataset
        net_all = X @ w                  # (N, n_outputs)
        y_all = step_function(net_all - theta)  # (N, n_outputs)
        rms = compute_rms(d2, y_all)
        rms_history.append(rms)

        # callback para la GUI / plotting / logger
        if callback is not None:
            try:
                #VALORES DE CADA ITERACION
                callback(epoch, rms, w.copy(), theta.copy())
            except Exception:
                # no queremos parar el entrenamiento si el callback falla
                pass

        if verbose:
            print(f"[Iter {epoch}/{max_iter}] RMS = {rms:.6f}")

        # condición de parada por RMS
        if rms <= epsilon:
            if verbose:
                print(f"Convergencia alcanzada (RMS={rms:.6f} <= ε={epsilon}). Iter {epoch}")
            break

    history = {
        "rms": rms_history,
        "w": w,
        "theta": theta,
        "iterations": iter_done
    }
    return history


# ---------------- Demo / utilidad para integrar con GUI ----------------
if __name__ == "__main__":
    # demo simple cuando se ejecuta el módulo: genera dataset 1 (3 inputs) y entrena
    import matplotlib.pyplot as plt
    from time import sleep

    # Generar dataset 1: 3 entradas binarias, regla: salida=1 si suma >= 2
    X_demo = np.array([[int(x) for x in f"{i:03b}"] for i in range(8)], dtype=float)  # (8,3)
    d_demo = (np.sum(X_demo, axis=1) >= 2).astype(float).reshape((-1, 1))            # (8,1)

    n_inputs = X_demo.shape[1]
    n_outputs = 1
    w0 = np.random.uniform(-1, 1, size=(n_inputs, n_outputs))
    theta0 = np.random.uniform(-1, 1, size=(n_outputs,))

    # Simple plot en tiempo real usando pyplot (solo demo en consola)
    fig, ax = plt.subplots()
    ax.set_xlabel("Iteración")
    ax.set_ylabel("RMS")
    ax.set_title("Entrenamiento (demo)")
    line_x = []
    line_y = []
    ln, = ax.plot(line_x, line_y, '-o')

    def demo_callback(it, rms, w, theta):
        line_x.append(it)
        line_y.append(rms)
        ln.set_data(line_x, line_y)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)  # pequeña pausa para refrescar

    print("Entrenando demo dataset 1 (3 entradas) ...")
    hist = train_perceptron(X_demo, d_demo, w0, theta0, eta=0.2, max_iter=500, epsilon=1e-6,
                            callback=demo_callback, verbose=True, shuffle=True)
    print("Entrenamiento terminado en iter:", hist["iterations"])
    plt.show()
