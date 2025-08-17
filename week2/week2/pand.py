import pandas as pd
import numpy as np

class DataExplorer:
    def __init__(self, filepath):
        """Load CSV data into a pandas DataFrame"""
        try:
            self.df = pd.read_csv(filepath)
            print(f"Data loaded successfully from: {filepath}\n")
        except FileNotFoundError:
            print(f"Error: File not found: {filepath}")
            self.df = None

    def summary_statistics(self):
        """Generate summary statistics for numeric and categorical columns"""
        if self.df is None:
            return None
        print("----- Summary Statistics -----\n")
        print(self.df.describe(include='all'))

    def missing_and_duplicates(self):
        """Detect missing values and duplicates"""
        if self.df is None:
            return None
        print("\n----- Missing Values -----")
        print(self.df.isnull().sum())

        print("\n----- Duplicate Rows -----")
        print(self.df.duplicated().sum())

    def detect_outliers(self):
        """Detect outliers using IQR method"""
        if self.df is None:
            return None
        print("\n----- Outliers (IQR method) -----")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_report = {}
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)]
            outlier_report[col] = len(outliers)
        for col, count in outlier_report.items():
            print(f"{col}: {count} outliers")

    def preprocessing_recommendations(self):
        """Provide simple preprocessing suggestions"""
        if self.df is None:
            return None
        print("\n----- Preprocessing Recommendations -----")
        if self.df.isnull().sum().sum() > 0:
            print("- Handle missing values (impute or drop)")
        else:
            print("- No missing values detected")

        if self.df.duplicated().sum() > 0:
            print("- Remove duplicate rows")
        else:
            print("- No duplicate rows detected")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                print(f"- Consider handling outliers in column: {col}")

    def explore(self):
        """Run all exploration steps"""
        if self.df is None:
            print("No data to explore.")
            return
        print("===== Data Exploration Report =====")
        print("\n----- Columns -----")
        print(self.df.columns.tolist())
        self.summary_statistics()
        self.missing_and_duplicates()
        self.detect_outliers()
        self.preprocessing_recommendations()


# -------- Usage Example --------
if __name__ == "__main__":
    # Replace with your CSV file path
    explorer = DataExplorer("employee_data.csv")
    explorer.explore()
