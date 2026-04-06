class ChartEngine:
    def __init__(self, df, column_info):
        self.df = df
        self.column_info = column_info

    def generate_charts(self):
        candidates = []

        numerical = self.column_info["numerical"]
        categorical = self.column_info["categorical"]
        datetime = self.column_info["datetime"]

        # 📈 Line Chart (time trend)
        if datetime and numerical:
            candidates.append({
                "type": "line",
                "x": datetime[0],
                "y": numerical[0],
                "score": 3,
                "reason": "Time trend detected"
            })

        # 📊 Bar Chart (category comparison)
        if categorical and numerical:
            col_cat = categorical[0]
            col_num = numerical[0]

            df_grouped = (
                self.df.groupby(col_cat)[col_num]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )

            candidates.append({
                "type": "bar",
                "x": col_cat,
                "y": col_num,
                "data": df_grouped.to_dict(),
                "score": 3,
                "reason": "Top category comparison (Top 10 applied)"
            })

        # 🥧 Pie Chart (only if few categories)
        if categorical and numerical:
            col_cat = categorical[0]

            if self.df[col_cat].nunique() < 6:
                candidates.append({
                    "type": "pie",
                    "names": col_cat,
                    "values": numerical[0],
                    "score": 2,
                    "reason": "Low category distribution"
                })

        # 📉 Histogram (distribution)
        if numerical:
            candidates.append({
                "type": "histogram",
                "x": numerical[0],
                "score": 2,
                "reason": "Distribution of values"
            })

        # 📊 Scatter (if 2 numeric columns)
        if len(numerical) >= 2:
            candidates.append({
                "type": "scatter",
                "x": numerical[0],
                "y": numerical[1],
                "score": 2,
                "reason": "Correlation analysis"
            })

        # 🔥 Sort charts by score
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

        # Return top 3 charts
        return candidates[:3]
    
        print("\nALL CANDIDATES:\n", candidates)