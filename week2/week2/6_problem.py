import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataTransformationPipeline:
    def __init__(self, df):
        self.df = df.copy()
        self.log = []

    # ---------------- Missing Value Handling ----------------
    def handle_missing(self, strategy="mean", columns=None, fill_value=None):
        """
        strategy: 'mean', 'median', 'mode', 'constant'
        columns: list of columns to fill, default all numeric
        fill_value: used if strategy='constant'
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if strategy == "mean":
                value = self.df[col].mean()
                self.df[col].fillna(value, inplace=True)
                self.log.append(f"Filled missing in {col} with mean: {value}")
            elif strategy == "median":
                value = self.df[col].median()
                self.df[col].fillna(value, inplace=True)
                self.log.append(f"Filled missing in {col} with median: {value}")
            elif strategy == "mode":
                value = self.df[col].mode()[0]
                self.df[col].fillna(value, inplace=True)
                self.log.append(f"Filled missing in {col} with mode: {value}")
            elif strategy == "constant":
                self.df[col].fillna(fill_value, inplace=True)
                self.log.append(f"Filled missing in {col} with constant: {fill_value}")
        return self

    # ---------------- Outlier Detection and Removal ----------------
    def remove_outliers(self, columns=None, method="zscore", threshold=3):
        """
        method: 'zscore' or 'iqr'
        threshold: z-score threshold
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if method == "zscore":
                mean = self.df[col].mean()
                std = self.df[col].std()
                condition = np.abs((self.df[col] - mean) / std) <= threshold
                removed = len(self.df) - sum(condition)
                self.df = self.df[condition]
                self.log.append(f"Removed {removed} outliers from {col} using z-score")
            elif method == "iqr":
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                condition = (self.df[col] >= Q1 - 1.5*IQR) & (self.df[col] <= Q3 + 1.5*IQR)
                removed = len(self.df) - sum(condition)
                self.df = self.df[condition]
                self.log.append(f"Removed {removed} outliers from {col} using IQR")
        return self

    # ---------------- Feature Scaling ----------------
    def scale_features(self, columns=None, method="standard"):
        """
        method: 'standard' (z-score) or 'minmax' (0-1 scaling)
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if method == "standard":
                scaler = StandardScaler()
                self.df[[col]] = scaler.fit_transform(self.df[[col]])
                self.log.append(f"Standard scaled column {col}")
            elif method == "minmax":
                scaler = MinMaxScaler()
                self.df[[col]] = scaler.fit_transform(self.df[[col]])
                self.log.append(f"MinMax scaled column {col}")
        return self

    # ---------------- Show Log ----------------
    def show_log(self):
        print("Transformation Log:")
        for entry in self.log:
            print(" -", entry)

# ---------------- Example Usage ----------------
if __name__ == "__main__":
    # Example dataset
    data = {
        'Age': [25, 30, 22, np.nan, 45, 120],
        'Salary': [50000, 60000, 52000, 58000, np.nan, 1000000]
    }
    df = pd.DataFrame(data)

    pipeline = DataTransformationPipeline(df)
    pipeline.handle_missing(strategy="mean")\
            .remove_outliers(method="zscore", threshold=3)\
            .scale_features(method="standard")
    
    pipeline.show_log()
    print("\nTransformed DataFrame:")
    print(pipeline.df)
