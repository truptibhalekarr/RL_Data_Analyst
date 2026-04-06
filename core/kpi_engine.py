import pandas as pd

class KPIEngine:
    def __init__(self, df, column_info):
        self.df = df
        self.column_info = column_info

    def generate_kpis(self):
        kpis = []

        numerical_cols = self.column_info["numerical"]
        datetime_cols = self.column_info["datetime"]
        categorical_cols = self.column_info["categorical"]

        # 🎯 1. Total (for first numeric column)
        if numerical_cols:
            col = numerical_cols[0]
            total = self.df[col].sum()

            kpis.append({
                "title": f"Total {col}",
                "value": float(total),
                "reason": "Aggregate sum of primary numerical column"
            })

        # 🎯 2. Average
        if numerical_cols:
            col = numerical_cols[0]
            avg = self.df[col].mean()

            kpis.append({
                "title": f"Average {col}",
                "value": float(avg),
                "reason": "Mean value of numerical data"
            })

        # 🎯 3. Top Category
        if categorical_cols and numerical_cols:
            cat_col = categorical_cols[0]
            num_col = numerical_cols[0]

            top_cat = (
                self.df.groupby(cat_col)[num_col]
                .sum()
                .idxmax()
            )

            kpis.append({
                "title": f"Top {cat_col}",
                "value": top_cat,
                "reason": "Highest contributing category"
            })

        # 🎯 4. Growth (if datetime exists)
        if datetime_cols and numerical_cols:
            date_col = datetime_cols[0]
            num_col = numerical_cols[0]

            df_sorted = self.df.sort_values(by=date_col)

            first = df_sorted[num_col].iloc[0]
            last = df_sorted[num_col].iloc[-1]

            if first != 0:
                growth = ((last - first) / first) * 100

                kpis.append({
                    "title": "Growth %",
                    "value": float(growth),
                    "reason": "Change over time detected"
                })

        return kpis