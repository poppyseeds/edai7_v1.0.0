"""Training-only mixed-table encoding; JSON state, no executable pickle preprocessing."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


class MixedTableCodec:
    def fit_transform(self, frame: pd.DataFrame):
        self.state = {"columns": list(frame.columns), "numerical": [], "categorical": [],
                      "dtypes": {c: str(frame[c].dtype) for c in frame}}
        nums, cats = [], []
        for column in frame:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series) and series.nunique() > 12:
                values = series.to_numpy(dtype=float, na_value=np.nan)
                median = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
                clean = np.nan_to_num(values, nan=median).reshape(-1, 1)
                transformer = QuantileTransformer(n_quantiles=min(1000, len(frame)), output_distribution="normal", random_state=0)
                nums.append(transformer.fit_transform(clean).ravel())
                self.state["numerical"].append({"column": column, "median": median,
                    "quantiles": transformer.quantiles_.ravel().tolist(), "references": transformer.references_.tolist()})
                if series.isna().any():
                    cats.append(series.isna().to_numpy(dtype=np.int64))
                    self.state["categorical"].append({"missing_indicator": column, "values": [False, True]})
            else:
                values = [None if pd.isna(v) else v.item() if isinstance(v, np.generic) else v for v in series]
                vocabulary = list(dict.fromkeys(values))
                mapping = {v: i for i, v in enumerate(vocabulary)}
                cats.append(np.array([mapping[v] for v in values]))
                self.state["categorical"].append({"column": column, "values": vocabulary})
        return (np.column_stack(nums).astype("float32") if nums else np.empty((len(frame), 0), dtype="float32"),
                np.column_stack(cats).astype("int64") if cats else np.empty((len(frame), 0), dtype="int64"))

    def inverse(self, numerical, categorical):
        result = {}
        for index, spec in enumerate(self.state["numerical"]):
            transformer = QuantileTransformer(n_quantiles=len(spec["references"]), output_distribution="normal")
            transformer.quantiles_ = np.array(spec["quantiles"]).reshape(-1, 1)
            transformer.references_ = np.array(spec["references"])
            transformer.n_quantiles_ = len(spec["references"])
            transformer.n_features_in_ = 1
            result[spec["column"]] = transformer.inverse_transform(numerical[:, index:index+1]).ravel()
        for index, spec in enumerate(self.state["categorical"]):
            values = np.array(spec["values"], dtype=object)[categorical[:, index]]
            if "missing_indicator" in spec:
                result[spec["missing_indicator"]][values.astype(bool)] = np.nan
            else:
                result[spec["column"]] = values
        frame = pd.DataFrame(result)[self.state["columns"]]
        for column, dtype in self.state["dtypes"].items():
            if dtype.startswith(("int", "uint", "Int", "UInt")):
                frame[column] = pd.to_numeric(frame[column]).round()
            try:
                frame[column] = frame[column].astype(dtype)
            except (ValueError, TypeError):
                pass
        return frame
