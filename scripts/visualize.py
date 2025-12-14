# src/eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display  # <--- THIS IS THE FIX

# Set style globally
sns.set_theme(style="whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')

def summarize_data(df: pd.DataFrame):
    """Prints shape, info, and summary stats."""
    print(f"Shape of the dataset: {df.shape}")
    print("-" * 30)
    print("\nData Info:")
    # .info() prints directly to stdout, so no display needed here
    df.info() 
    print("-" * 30)
    
    print("\nNumerical Stats:")
    # display() now works because we imported it
    display(df.describe()) 
    
    print("\nCategorical Stats:")
    display(df.describe(include=['object']))

def plot_numerical_distributions(df: pd.DataFrame, columns: list):
    """Plots histograms for numerical columns."""
    plt.figure(figsize=(15, 6))
    for i, col in enumerate(columns, 1):
        plt.subplot(1, len(columns), i)
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_categorical_distributions(df: pd.DataFrame, columns: list):
    """Plots bar charts for categorical columns."""
    plt.figure(figsize=(18, 12))
    for i, col in enumerate(columns, 1):
        plt.subplot(2, 2, i)
        
        counts = df[col].value_counts()
        if len(counts) > 10:
            counts = counts.head(10)
            title_suffix = " (Top 10)"
        else:
            title_suffix = ""
            
        sns.barplot(x=counts.index, y=counts.values, palette='viridis')
        plt.title(f'Distribution of {col}{title_suffix}')
        plt.xticks(rotation=45)
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame):
    """Plots heatmap for numerical correlations."""
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

def check_missing_values(df: pd.DataFrame):
    """Identifies and plots missing values."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        missing_pct = (missing / len(df)) * 100
        print("Columns with missing values:")
        # Display the series nicely
        display(missing_pct) 
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_pct.index, y=missing_pct.values, palette='Reds_r')
        plt.title('Percentage of Missing Values')
        plt.show()
    else:
        print("No missing values found.")

def plot_outliers(df: pd.DataFrame, numerical_cols: list, target_col: str):
    """Plots boxplots to detect outliers vs the target."""
    plt.figure(figsize=(15, 6))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(1, 2, i)
        sns.boxplot(y=df[col], x=df[target_col].astype(str))
        plt.title(f'Box Plot of {col} by {target_col}')
        plt.yscale('log') 
        plt.xlabel(f'{target_col}')
    plt.tight_layout()
    plt.show()