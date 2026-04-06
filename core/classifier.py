import pandas as pd

class DataClassifier:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def classify_columns(self):
        numerical_cols = []
        categorical_cols = []
        datetime_cols = []

        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                numerical_cols.append(col)

            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                datetime_cols.append(col)

            else:
                try:
                    converted = pd.to_datetime(self.df[col], errors='raise')
                    datetime_cols.append(col)
                except:
                    categorical_cols.append(col)

        return {
            "numerical": numerical_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols
        }

    def generate_summary(self):
        summary = {}

        for col in self.df.columns:
            col_data = self.df[col]

            summary[col] = {
                "missing_values": int(col_data.isnull().sum()),
                "unique_values": int(col_data.nunique())
            }

            if pd.api.types.is_numeric_dtype(col_data):
                summary[col].update({
                    "mean": float(col_data.mean()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max())
                })

            # Bonus: low cardinality detection
            if col_data.nunique() < 10:
                summary[col]["low_cardinality"] = True

        return summary

    def analyze(self):
        return {
            "column_types": self.classify_columns(),
            "summary": self.generate_summary()
        }