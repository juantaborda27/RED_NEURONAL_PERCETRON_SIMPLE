# src/data_utils.py
import pandas as pd
import json
from pathlib import Path

def load_simple(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
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
        n_inputs = n_columns - 1
        n_outputs = 1
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
