import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import WOEEncoder

def load_data(filepath):
    """Loads raw data from CSV and maps specific columns."""
    df = pd.read_csv(filepath)
    
    # --- MAPPING YOUR SPECIFIC COLUMNS ---
    # We rename your columns to match the standard names used in the pipeline
    df = df.rename(columns={
        'TransactionStartTime': 'TransactionDate',
        'ChannelId': 'Channel'
    })

    # Convert Date
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    
    # Ensure Amount is positive (handling potential negative values)
    df['Amount'] = df['Amount'].abs()
    
    return df

def create_rfm_features(df):
    """Task 4: Calculates RFM metrics and creates Proxy Label."""
    # Create a snapshot date (1 day after the last transaction)
    snapshot_date = df['TransactionDate'].max() + pd.Timedelta(days=1)

    # Calculate Recency, Frequency, Monetary
    rfm = df.groupby('CustomerId').agg({
        'TransactionDate': lambda x: (snapshot_date - x.max()).days, # Recency
        'TransactionId': 'count',                                     # Frequency
        'Amount': 'sum'                                               # Monetary
    }).reset_index()

    rfm.rename(columns={
        'TransactionDate': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    }, inplace=True)

    # Scaling for Clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # Clustering (Task 4)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Define High Risk: 
    # Logic: The cluster with the LOWEST mean Monetary value is treated as "High Risk" (Low engagement/Low value)
    cluster_summary = rfm.groupby('Cluster')['Monetary'].mean()
    high_risk_cluster = cluster_summary.idxmin()

    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    
    return rfm[['CustomerId', 'Recency', 'Frequency', 'Monetary', 'is_high_risk']]

def aggregate_features(df):
    """Task 3: Aggregates transactional data to customer level."""
    
    # Extract temporal features
    df['TransactionHour'] = df['TransactionDate'].dt.hour
    df['TransactionDay'] = df['TransactionDate'].dt.day
    
    # Numerical Aggregations
    num_aggs = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'std'],
        'TransactionHour': 'mean', 
        'TransactionDay': 'mean'
    })
    # Flatten MultiIndex columns (e.g., Amount_sum, Amount_mean)
    num_aggs.columns = ['_'.join(col).strip() for col in num_aggs.columns.values]
    
    # Categorical: Mode (Most frequent category per customer)
    # Using 'ProductCategory' and 'Channel' (renamed from ChannelId)
    cat_aggs = df.groupby('CustomerId')[['ProductCategory', 'Channel']].agg(lambda x: x.mode()[0])
    
    final_df = pd.concat([num_aggs, cat_aggs], axis=1).reset_index()
    return final_df

def build_preprocessor(numeric_features, categorical_features):
    """Task 3: Builds the Scikit-Learn Pipeline for processing."""
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Using Weight of Evidence (WoE) for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('woe', WOEEncoder()) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def prepare_training_data(raw_filepath):
    """Main orchestration function."""
    # 1. Load Data
    raw_df = load_data(raw_filepath)
    
    # 2. Create Target (RFM)
    labels = create_rfm_features(raw_df)
    
    # 3. Create Features (Aggregates)
    features = aggregate_features(raw_df)
    
    # 4. Merge Features and Labels
    full_data = pd.merge(features, labels, on='CustomerId')
    
    # Drop CustomerId as it's an identifier, not a feature
    full_data = full_data.drop(columns=['CustomerId'])
    
    return full_data