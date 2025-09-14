# src/main.py
"""
Interfaz mínima para Perceptrón unicapa:
- Botones: ENTRENAMIENTO, SIMULACION
- En ENTRENAMIENTO: seleccionar dataset, mostrar entradas/salidas/patrones,
  inicializar parámetros aleatorios (pesos y umbral) y mostrar valores.
Requisitos: python 3.8+, numpy, pandas, tkinter (tk viene con Python).
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import pandas as pd
import numpy as np
import json
import os


import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from trainer import train_perceptron

# Carpeta por defecto donde están los datasets (relativa al proyecto)
DATASETS_FOLDER = Path("datasets")


# ---------- Utilities de carga (muy simple, sin limpieza) ----------
def load_simple(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext == ".json":
        # pd.read_json suele manejar listas de objetos/estructuras tabulares
        try:
            return pd.read_json(path)
        except ValueError:
            # fallback: leer con json + normalizar
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.json_normalize(data)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=0)
    else:
        raise ValueError("Extensión no soportada")


def dataset_summary(df: pd.DataFrame):
    n_patterns = df.shape[0]
    n_columns = df.shape[1]
    if n_columns == 0:
        n_inputs, n_outputs = 0, 0
    elif n_columns == 1:
        n_inputs, n_outputs = 0, 1
    else:
        n_inputs = n_columns - 1  # asumimos última columna target
        n_outputs = 1
    # columnas de entrada y nombre del target (última columna)
    cols = list(df.columns)
    input_cols = cols[:-1] if len(cols) > 1 else []
    target_col = cols[-1] if len(cols) >= 1 else None
    return {
        "patrones": n_patterns,
        "entradas": n_inputs,
        "salidas": n_outputs,
        "input_cols": input_cols,
        "target_col": target_col,
        "columns": cols
    }


# ---------- GUI ----------
class PerceptronApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perceptrón Unicapa - Interfaz mínima")
        self.geometry("900x600")
        self.resizable(True, True)

        # frames principales
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        body_frame = tk.Frame(self)
        body_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        # botones principales
        self.bt_entrenamiento = tk.Button(top_frame, text="ENTRENAMIENTO",
                                         width=20, height=2, command=self.show_entrenamiento)
        self.bt_entrenamiento.pack(side=tk.LEFT, padx=10)

        self.bt_simulacion = tk.Button(top_frame, text="SIMULACIÓN",
                                       width=20, height=2, command=self.show_simulacion)
        self.bt_simulacion.pack(side=tk.LEFT, padx=10)

        # area donde se intercambia el contenido
        self.content = tk.Frame(body_frame)
        self.content.pack(fill=tk.BOTH, expand=True)

        # frames para los modos
        self.entreno_frame = None
        self.simula_frame = None

        # estado
        self.current_df = None
        self.current_path = None
        self.summary = None
        self.weights = None
        self.theta = None

        # iniciar mostrando entrenamiento por defecto
        self.show_entrenamiento()

    # ---------- Vistas ----------
    def clear_content(self):
        for widget in self.content.winfo_children():
            widget.destroy()

    def show_simulacion(self):
        self.clear_content()
        frame = tk.Frame(self.content)
        frame.pack(fill=tk.BOTH, expand=True)
        lbl = tk.Label(frame, text="Módulo SIMULACIÓN (vacío por ahora)", font=("Arial", 14))
        lbl.pack(pady=20)
        self.simula_frame = frame

    def show_entrenamiento(self):
        self.clear_content()
        frame = tk.Frame(self.content)
        frame.pack(fill=tk.BOTH, expand=True)

        # Left panel: selección dataset + info
        left = tk.Frame(frame)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        tk.Label(left, text="Seleccionar dataset:", font=("Arial", 11)).pack(anchor="w")
        # Combobox con archivos encontrados
        self.dataset_combobox = ttk.Combobox(left, state="readonly", width=40)
        self.dataset_combobox.pack(anchor="w", pady=4)
        files = self.scan_datasets()
        self.dataset_combobox["values"] = files
        if files:
            self.dataset_combobox.current(0)
        self.dataset_combobox.bind("<<ComboboxSelected>>", self.on_dataset_selected)

        # Opción para cargar file arbitrario
        tk.Button(left, text="Cargar otro archivo...", command=self.load_other_file).pack(anchor="w", pady=6)

        # Info simple (entradas/salidas/patrones)
        info_frame = tk.LabelFrame(left, text="Resumen", padx=6, pady=6)
        info_frame.pack(fill=tk.X, pady=6)
        self.lbl_patrones = tk.Label(info_frame, text="Patrones: -")
        self.lbl_patrones.pack(anchor="w")
        self.lbl_entradas = tk.Label(info_frame, text="Entradas: -")
        self.lbl_entradas.pack(anchor="w")
        self.lbl_salidas = tk.Label(info_frame, text="Salidas: -")
        self.lbl_salidas.pack(anchor="w")
        self.lbl_cols = tk.Label(info_frame, text="Columnas: -", wraplength=280, justify="left")
        self.lbl_cols.pack(anchor="w", pady=(4,0))

        # Preview de las primeras filas
        preview_frame = tk.LabelFrame(left, text="Preview (5 primeros)", padx=6, pady=6)
        preview_frame.pack(fill=tk.BOTH, expand=False, pady=6)
        self.txt_preview = tk.Text(preview_frame, height=8, width=50)
        self.txt_preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.txt_preview.configure(state="disabled")

        # Right panel: parámetros e inicialización
        right = tk.Frame(frame)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        params_frame = tk.LabelFrame(right, text="Configuración (editar si quiere)", padx=6, pady=6)
        params_frame.pack(fill=tk.X, pady=6)

        tk.Label(params_frame, text="Tasa de aprendizaje η (0<η≤1):").grid(row=0, column=0, sticky="w")
        self.entry_eta = tk.Entry(params_frame, width=10)
        self.entry_eta.insert(0, "0.1")
        self.entry_eta.grid(row=0, column=1, sticky="w", padx=6, pady=2)

        tk.Label(params_frame, text="Max iteraciones:").grid(row=1, column=0, sticky="w")
        self.entry_maxiter = tk.Entry(params_frame, width=10)
        self.entry_maxiter.insert(0, "1000")
        self.entry_maxiter.grid(row=1, column=1, sticky="w", padx=6, pady=2)

        tk.Label(params_frame, text="Error máximo ε:").grid(row=2, column=0, sticky="w")
        self.entry_epsilon = tk.Entry(params_frame, width=10)
        self.entry_epsilon.insert(0, "0.01")
        self.entry_epsilon.grid(row=2, column=1, sticky="w", padx=6, pady=2)

        # Inicializar parámetros
        init_frame = tk.Frame(right)
        init_frame.pack(fill=tk.X, pady=8)
        self.bt_init = tk.Button(init_frame, text="Inicializar parámetros aleatorios",
                                 command=self.initialize_parameters, state="disabled")
        self.bt_init.pack(anchor="w")

        # Mostrar pesos y theta
        weights_frame = tk.LabelFrame(right, text="Pesos y umbral (θ)", padx=6, pady=6)
        weights_frame.pack(fill=tk.BOTH, expand=True, pady=6)
        self.txt_weights = tk.Text(weights_frame, height=15)
        self.txt_weights.pack(fill=tk.BOTH, expand=True)
        self.txt_weights.configure(state="disabled")

        # Guardar botón (por si quiere exportar)
        save_frame = tk.Frame(right)
        save_frame.pack(fill=tk.X, pady=4)
        tk.Button(save_frame, text="Guardar pesos a JSON", command=self.save_weights).pack(side=tk.LEFT)

        self.entreno_frame = frame

        # Si había archivos, seleccionar el primero por defecto
        if files:
            # for initial load fire selection
            self.on_dataset_selected()

    # ---------- acciones ----------
    def scan_datasets(self):
        p = DATASETS_FOLDER
        if not p.exists():
            return []
        files = []
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in {".csv", ".json", ".xlsx", ".xls"}:
                files.append(str(f))
        return files

    def load_other_file(self):
        file = filedialog.askopenfilename(title="Seleccionar dataset",
                                          filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json"), ("Excel files", "*.xlsx;*.xls")])
        if file:
            # si quiere, añadimos a combobox (no persistimos)
            vals = list(self.dataset_combobox["values"])
            if file not in vals:
                vals.append(file)
                self.dataset_combobox["values"] = vals
            idx = self.dataset_combobox["values"].index(file)
            self.dataset_combobox.current(idx)
            self.on_dataset_selected()

    def on_dataset_selected(self, event=None):
        sel = self.dataset_combobox.get()
        if not sel:
            return
        try:
            df = load_simple(Path(sel))
        except Exception as e:
            messagebox.showerror("Error al cargar", f"No se pudo leer el archivo:\n{e}")
            return

        self.current_df = df
        self.current_path = sel
        self.summary = dataset_summary(df)
        # actualizar labels
        self.lbl_patrones.config(text=f"Patrones: {self.summary['patrones']}")
        self.lbl_entradas.config(text=f"Entradas: {self.summary['entradas']}")
        self.lbl_salidas.config(text=f"Salidas: {self.summary['salidas']}")
        cols_text = ", ".join(self.summary["columns"])
        self.lbl_cols.config(text=f"Columnas: {cols_text}")

        # preview
        self.txt_preview.configure(state="normal")
        self.txt_preview.delete("1.0", tk.END)
        preview_df = df.head(5)
        self.txt_preview.insert(tk.END, preview_df.to_string(index=False))
        self.txt_preview.configure(state="disabled")

        # habilitar inicialización
        self.bt_init.configure(state="normal")
        # limpiar pesos previos
        self.weights = None
        self.theta = None
        self._display_weights()

    def initialize_parameters(self):
        if self.summary is None:
            messagebox.showwarning("Dataset no seleccionado", "Seleccione primero un dataset.")
            return
        n_inputs = self.summary["entradas"]
        n_outputs = self.summary["salidas"]  # asumimos 1 por defecto, puede ser >1 en general

        # CORRECCIÓN: pesos w con forma (n_inputs, n_outputs)
        # Cada columna de w corresponde a los pesos para una salida.
        if n_inputs <= 0:
            # caso borde: sin entradas
            self.weights = np.zeros((0, n_outputs))
        else:
            self.weights = np.random.uniform(-1.0, 1.0, size=(n_inputs, n_outputs))

        # theta: umbral por salida (vector length = n_outputs)
        self.theta = np.random.uniform(-1.0, 1.0, size=(n_outputs,))

        # mostrar en el text widget
        self._display_weights()

    def _display_weights(self):
        self.txt_weights.configure(state="normal")
        self.txt_weights.delete("1.0", tk.END)
        if self.weights is None:
            self.txt_weights.insert(tk.END, "Pesos no inicializados.\nPulse 'Inicializar parámetros aleatorios'.")
        else:
            self.txt_weights.insert(tk.END, f"Pesos (w) shape: {self.weights.shape}  (n_inputs x n_outputs)\n\n")
            n_inputs, n_outputs = self.weights.shape
            # Mostrar por salida (columna)
            for out_idx in range(n_outputs):
                col = self.weights[:, out_idx]
                self.txt_weights.insert(tk.END, f"Salida {out_idx} (pesos para cada entrada): {np.array2string(col, precision=4)}\n")
            self.txt_weights.insert(tk.END, f"\nUmbral θ (por salida): {np.array2string(self.theta, precision=4)}\n")
            # mostrar también valores de η, max_iter, epsilon actuales
            eta = self.entry_eta.get().strip() if self.entry_eta.get().strip() else "0.1"
            maxiter = self.entry_maxiter.get().strip() if self.entry_maxiter.get().strip() else "1000"
            eps = self.entry_epsilon.get().strip() if self.entry_epsilon.get().strip() else "0.01"
            self.txt_weights.insert(tk.END, f"\nParámetros actuales:\n η = {eta}\n max_iter = {maxiter}\n ε = {eps}\n")
        self.txt_weights.configure(state="disabled")

    def save_weights(self):
        if self.weights is None:
            messagebox.showinfo("Sin pesos", "No hay pesos para guardar. Inicialice primero.")
            return
        # pedir ruta para guardar
        fpath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")],
                                             title="Guardar pesos como...")
        if not fpath:
            return
        data = {
            "weights": self.weights.tolist(),
            "theta": self.theta.tolist(),
            "eta": float(self.entry_eta.get()),
            "max_iter": int(self.entry_maxiter.get()),
            "epsilon": float(self.entry_epsilon.get()),
            "dataset": str(self.current_path)
        }
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Guardado", f"Pesos guardados en:\n{fpath}")
        except Exception as e:
            messagebox.showerror("Error al guardar", str(e))


# ---------- Run ----------
def main():
    # ensure datasets folder exists (no se requiere que tenga archivos)
    if not DATASETS_FOLDER.exists():
        try:
            os.makedirs(DATASETS_FOLDER)
        except Exception:
            pass

    app = PerceptronApp()
    app.mainloop()


if __name__ == "__main__":
    main()





