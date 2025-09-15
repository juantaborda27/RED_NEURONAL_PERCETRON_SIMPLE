import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import json

class SimulationFrame(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        # estado
        self.weights = None
        self.theta = None
        self.entry_inputs = []

        # Botón cargar archivo
        tk.Button(self, text="Cargar archivo de entrenamiento (JSON)", command=self.load_json_file).pack(anchor="w", pady=5)

        # Info de parámetros
        self.txt_params = tk.Text(self, height=8, width=70)
        self.txt_params.pack(fill=tk.X, pady=5)
        self.txt_params.configure(state="disabled")

        # Entradas dinámicas
        self.inputs_frame = tk.LabelFrame(self, text="Ingresar patrón", padx=6, pady=6)
        self.inputs_frame.pack(fill=tk.X, pady=5)

        # Botón calcular
        self.bt_calc = tk.Button(self, text="Calcular salida", state="disabled", command=self.calculate_output)
        self.bt_calc.pack(pady=5)

        # Resultado
        self.lbl_result = tk.Label(self, text="Resultado: -", font=("Arial", 12))
        self.lbl_result.pack(pady=10)

    def load_json_file(self):
        file_path = filedialog.askopenfilename(
            title="Seleccione el archivo JSON del entrenamiento",
            filetypes=[("JSON files", "*.json")]
        )
        if not file_path:
            return

        with open(file_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        last_iter = history[-1]
        self.weights = np.array(last_iter["weights"]).flatten()
        self.theta = np.array(last_iter["theta"]).flatten()

        # Mostrar info
        self.txt_params.configure(state="normal")
        self.txt_params.delete("1.0", tk.END)
        self.txt_params.insert(tk.END, f"Iteración: {last_iter['iter']}\n")
        self.txt_params.insert(tk.END, f"RMS: {last_iter['rms']:.4f}\n")
        self.txt_params.insert(tk.END, f"Pesos: {self.weights}\n")
        self.txt_params.insert(tk.END, f"Umbral θ: {self.theta}\n")
        self.txt_params.configure(state="disabled")

        # Crear campos de entrada
        for widget in self.inputs_frame.winfo_children():
            widget.destroy()

        self.entry_inputs = []
        for i in range(len(self.weights)):
            tk.Label(self.inputs_frame, text=f"x{i+1}:").grid(row=0, column=i*2, padx=4, pady=4)
            entry = tk.Entry(self.inputs_frame, width=8)
            entry.grid(row=0, column=i*2+1, padx=4, pady=4)
            self.entry_inputs.append(entry)

        self.bt_calc.configure(state="normal")

    def calculate_output(self):
        try:
            x = np.array([float(e.get()) for e in self.entry_inputs])
        except ValueError:
            messagebox.showerror("Error", "Ingrese valores numéricos válidos para las entradas.")
            return

        # Asegurar que pesos y theta sean arrays planos
        w = np.array(self.weights).flatten()
        t = np.array(self.theta).flatten()

        u = np.dot(x, w) - t
        u = float(u)  # convertir a escalar

        y = 1 if u >= 0 else 0

        self.lbl_result.config(
            text=f"Entrada: {x}\nPotencial neto u = {u:.4f}\nSalida de la red = {y}"
        )

