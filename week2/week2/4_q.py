import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class DashboardVisualizer:
    def __init__(self, filename):
        """Load CSV data into DataFrame, searching parent folders if needed"""
        self.df = None
        self.filepath = self.find_file(filename)
        if self.filepath:
            try:
                self.df = pd.read_csv(self.filepath)
                print(f"Data loaded from: {self.filepath}")
            except Exception as e:
                print(f"Error loading file: {e}")
        else:
            print(f"File not found: {filename}")

    def find_file(self, filename):
        """Search current folder and parent folders for the file"""
        folder = os.getcwd()
        for _ in range(3):  # search current + 2 parent folders
            candidate = os.path.join(folder, filename)
            if os.path.exists(candidate):
                return candidate
            folder = os.path.dirname(folder)
        return None

    def auto_chart_type(self, col):
        """Determine chart type based on column data type"""
        if self.df[col].dtype in ['int64', 'float64']:
            return 'hist'
        else:
            return 'count'

    def create_dashboard(self, max_cols=4):
        """Create multi-panel dashboard for all columns"""
        if self.df is None:
            print("No data to visualize.")
            return
        
        cols = self.df.columns
        n_cols = min(len(cols), max_cols)
        n_rows = int(np.ceil(len(cols) / n_cols))
        
        plt.figure(figsize=(5*n_cols, 4*n_rows))
        plt.suptitle("Multi-Panel Data Dashboard", fontsize=16)
        
        for i, col in enumerate(cols, 1):
            plt.subplot(n_rows, n_cols, i)
            chart_type = self.auto_chart_type(col)
            
            if chart_type == 'hist':
                sns.histplot(self.df[col], kde=True, color='skyblue')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.title(f'Histogram of {col}')
                # annotate mean
                mean_val = self.df[col].mean()
                plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                plt.legend()
            else:
                sns.countplot(x=col, data=self.df, palette='Set2')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.title(f'Countplot of {col}')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


# ---------------- Usage Example ----------------
if __name__ == "__main__":
    # Just give the CSV filename, the script will find it
    visualizer = DashboardVisualizer("employee_data.csv")
    visualizer.create_dashboard()
