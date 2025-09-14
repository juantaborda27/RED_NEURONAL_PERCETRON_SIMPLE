# src/simple_summary.py
from pathlib import Path
import pandas as pd

def simple_summary(path: str):
    """
    Lee dataset (csv, json, xlsx) y muestra:
      - patrones (número de filas)
      - entradas (número de columnas - 1, se asume la última columna como salida)
      - salidas (1)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    ext = p.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(p)
    elif ext == ".json":
        # pd.read_json maneja listas de objetos o tablas simples
        df = pd.read_json(p)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        raise ValueError(f"Extensión no soportada: {ext}")

    n_patterns = df.shape[0]
    n_columns = df.shape[1]

    if n_columns == 0:
        n_inputs = 0
        n_outputs = 0
    elif n_columns == 1:
        n_inputs = 0
        n_outputs = 1
    else:
        n_inputs = n_columns - 1
        n_outputs = 1  # asumimos la última columna como target

    return {"file": str(p), "patrones": n_patterns, "entradas": n_inputs, "salidas": n_outputs}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python src/simple_summary.py <ruta_al_dataset>")
        sys.exit(1)
    info = simple_summary(sys.argv[1])
    print(f"Archivo: {info['file']}")
    print(f"Patrones: {info['patrones']}")
    print(f"Entradas: {info['entradas']}")
    print(f"Salidas: {info['salidas']}")
