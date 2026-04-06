import pandas as pd

class InsightEngine:
    def __init__(self, df, column_info):
        self.df = df
        self.column_info = column_info

    def generate_insights(self):
        insights = []

        numerical = self.column_info["numerical"]
        categorical = self.column_info["categorical"]
        datetime = self.column_info["datetime"]

        # 📈 Trend Insight
        if datetime and numerical:
            date_col = datetime[0]
            num_col = numerical[0]

            df_sorted = self.df.sort_values(by=date_col)

            first = df_sorted[num_col].iloc[0]
            last = df_sorted[num_col].iloc[-1]

            if first != 0:
                change = ((last - first) / first) * 100

                if change > 0:
                    insights.append(f"{num_col} increased by {round(change,2)}% over time")
                else:
                    insights.append(f"{num_col} decreased by {round(abs(change),2)}% over time")

        # 🏆 Top Category Insight
        if categorical and numerical:
            cat_col = categorical[0]
            num_col = numerical[0]

            grouped = self.df.groupby(cat_col)[num_col].sum()

            top = grouped.idxmax()

            insights.append(f"{top} is the highest contributing {cat_col}")

        # 📊 Contribution Insight
        if categorical and numerical:
            grouped = self.df.groupby(categorical[0])[numerical[0]].sum()
            total = grouped.sum()

            top3 = grouped.sort_values(ascending=False).head(3).sum()

            percent = (top3 / total) * 100

            insights.append(f"Top 3 {categorical[0]} contribute {round(percent,2)}% of total {numerical[0]}")

        # 🚨 Anomaly (simple)
        if numerical:
            col = numerical[0]

            mean = self.df[col].mean()
            max_val = self.df[col].max()

            if max_val > mean * 1.5:
                insights.append(f"Unusually high peak detected in {col}")

        return insights