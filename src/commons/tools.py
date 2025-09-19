import os
from scipy import stats
import pandas as pd

input_path = os.path.join('..', 'input')
output_path = os.path.join('..', 'output')

def check_directories():
    paths = [
        input_path, output_path,
        os.path.join(output_path, 'databases'),
        os.path.join(output_path, 'descriptive_mining'),
        os.path.join(output_path, 'predictive_mining'),
        os.path.join(output_path, 'predictive_mining', 'train_set'),
        os.path.join(output_path, 'predictive_mining', 'deploy_set')
    ]
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

def input_numeric_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if -0.5<stats.skew(df[col].dropna())<0.5:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].median())
    return df