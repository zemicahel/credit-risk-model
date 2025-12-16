import pytest
import pandas as pd
import numpy as np
from src.data_processing import aggregate_features, create_rfm_features

@pytest.fixture
def sample_data():
    data = {
        'TransactionId': [1, 2, 3, 4, 5],
        'CustomerId': [101, 101, 102, 102, 103],
        'TransactionDate': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-05', '2023-01-10', '2023-01-20']),
        'Amount': [100, 200, 50, 150, 300],
        'ProductCategory': ['A', 'A', 'B', 'B', 'C'],
        'Channel': ['Online', 'Online', 'App', 'App', 'Store']
    }
    return pd.DataFrame(data)

def test_aggregate_features_shape(sample_data):
    """Test if aggregation returns one row per customer."""
    result = aggregate_features(sample_data)
    # We have 3 unique customers (101, 102, 103)
    assert len(result) == 3
    # Check if aggregation columns exist
    assert 'Amount_sum' in result.columns
    assert 'Amount_mean' in result.columns

def test_create_rfm_features(sample_data):
    """Test if RFM calculation and labelling works."""
    result = create_rfm_features(sample_data)
    
    # Check columns
    expected_cols = ['CustomerId', 'Recency', 'Frequency', 'Monetary', 'is_high_risk']
    for col in expected_cols:
        assert col in result.columns
    
    # Check if is_high_risk is binary
    assert set(result['is_high_risk'].unique()).issubset({0, 1})
    
    # Check logic: Customer 101 spent 300 total
    c101 = result[result['CustomerId'] == 101].iloc[0]
    assert c101['Monetary'] == 300