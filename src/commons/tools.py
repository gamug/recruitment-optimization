import datetime, os
from scipy import stats
import pandas as pd, numpy as np
from sklearn.impute import KNNImputer

input_path = os.path.join('..', 'input')
output_path = os.path.join('..', 'output')
cols_high_correlated = [
    'Desc_Cargo_AYUDANTE DE OBRA tasa 6.96', 'CTNENCUEST', 'TP16_HOG',
    'TVIVIENDA', 'TP9_2_2_MI', 'TP19_RECB1', 'TP19_INTE2', 'TP19_EE_1',
    'TP19_ALC_1', 'TP19_INTE9', 'TP15_4_OCU', 'TP19_RECB2'
]

def check_directories() -> None:
    '''Check if the required directories exist, if not create them'''
    paths = [
        input_path, output_path,
        os.path.join(output_path, 'databases'),
        os.path.join(output_path, 'models'),
        os.path.join(output_path, 'descriptive_mining'),
        os.path.join(output_path, 'predictive_mining'),
        os.path.join(output_path, 'predictive_mining', 'train_set'),
        os.path.join(output_path, 'predictive_mining', 'deploy_set')
    ]
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

def input_numeric_col(df: pd.DataFrame, col: str='knn') -> pd.DataFrame:
    '''Input missing values in numeric columns
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numeric columns
    col : str, optional
        Column to input missing values. If 'knn' input all columns using KNNImputer, by default 'knn'
    Returns
    -------
    pd.DataFrame
        DataFrame with inputed missing values'''
    assert col in df.columns.tolist() or col=='knn', 'Column not in DataFrame'
    if col!='knn':
        if -0.5<stats.skew(df[col].dropna())<0.5:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].median())
    else:
        imputer = KNNImputer(n_neighbors=3)
        df_imputed = imputer.fit_transform(df)
        df = pd.DataFrame(df_imputed, columns=df.columns)
    return df

def years_computing(dataset: pd.DataFrame) -> pd.DataFrame:
    '''Compute years from date of birth and clean some values
    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame with date of birth column
    Returns
    -------
    pd.DataFrame
        DataFrame with years column and cleaned values'''
    desc_cargo_eq = {
        "CONDUCTOR VOLQUETA DAF": "CONDUCTOR DE VOLQUETA DAF",
        "AUXILIAR ADMINISTRATIVA": "AUXILIAR ADMINSTRATIVO",
        "INSPECTOR SST": "INSPECTOR SST I",
        "SOLDADOR ": "SOLDADOR I"
    }
    dataset['Desc_Cargo'] = dataset['Desc_Cargo'].replace(desc_cargo_eq)
    dataset_ = dataset.copy()
    dataset_.insert(10, 'anios', (datetime.datetime.now()-dataset.fecha_nacimiento).dt.days//365.25)
    dataset_ = dataset_.drop('fecha_nacimiento', axis=1)
    dataset_ = dataset_[~(dataset_.causa_retiro=='MUERTE DEL TRABAJADOR')]
    return dataset_

def feature_dane(df: pd.DataFrame) -> pd.DataFrame:
    '''Feature engineering for DANE columns. It divides for total feature count in the fields of
    Persons, Houses, Surveys and Homes. This is done to avoid high correlation between these 
    variables and the rest of the features.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DANE columns. 
    Returns
    -------
    pd.DataFrame
        DataFrame with featured DANE columns'''
    total_counting = {
        'TP27_PERSO': 'persons', #número total de personas
        'TVIVIENDA': 'houses', #conteo de viviendas
        'CTNENCUEST': 'surveys', #cantidad de encuestas
        'TP16_HOG': 'homes'
    }
    feature_bars = {
        'TP27_PERSO': [
            'TP51_13_ED', 'TP51SUPERI', 'TP51SECUND', 'TP51PRIMAR', 'TP51_99_ED', 'TP34_6_EDA',
            'TP34_8_EDA', 'TP34_7_EDA', 'TP34_3_EDA', 'TP34_5_EDA', 'TP34_9_EDA', 'TP34_4_EDA',
            'TP34_2_EDA', 'TP34_1_EDA', 'TP32_1_SEX', 'TP32_2_SEX', 'TP51POSTGR'
        ],
        'TVIVIENDA': [
            'TP9_1_USO', 'TP19_INTE1', 'TP19_GAS_1', 'TP19_ACU_1', 'TP19_GAS_9',
            'TP19_EE_E2', 'TP19_EE_E3', 'TP19_EE_E5', 'TP19_EE_E6', 'TP15_1_OCU',
            'TP14_2_TIP', 'TP9_2_USO', 'TP14_6_TIP', 'TP15_2_OCU', 'TP14_4_TIP',
        ],
        'CTNENCUEST': ['TP4_2_NO', 'TP3_2_NO'],
        'TP16_HOG': ['TP27_PERSO']
    }
    featured_dataset, drop_vars = df.copy(), []
    for key, value in feature_bars.items():
        for var in value:
            featured_dataset[f'{total_counting[key]}_{var}'] = featured_dataset[var]/featured_dataset[key]
        drop_vars.extend(value)
    featured_dataset = input_numeric_col(featured_dataset)
    featured_dataset = featured_dataset.drop(np.unique(drop_vars), axis=1)
    return featured_dataset

def outliers_remotion(dataset_: pd.DataFrame) -> pd.DataFrame:
    '''Remove outliers in the years column
    Parameters
    ----------
    dataset_ : pd.DataFrame
        DataFrame with years column
    Returns
    -------
    pd.DataFrame
        DataFrame without outliers in years column'''
    dataset_.loc[dataset_.anios<18, 'anios'] = np.nan
    dataset_.loc[dataset_.anios>60, 'anios'] = np.nan
    dataset_ = input_numeric_col(dataset_, 'anios')
    return dataset_

def get_dummies(dataset_: pd.DataFrame, cat_cols: list, labeling_scope: bool=True) -> pd.DataFrame:
    '''Get dummies for categorical columns and encode objective variable
    Parameters
    ----------
    dataset_ : pd.DataFrame
        DataFrame with categorical columns
    cat_cols : list
        List with categorical columns
    labeling_scope : bool, optional
        If True, encode objective variable for scope analysis, by default True
    Returns
    -------
    pd.DataFrame
        DataFrame with dummies and encoded objective variable'''
    cat_dataset = dataset_[cat_cols]
    cat_dataset = cat_dataset.astype({var: str for var in cat_cols})
    numeric_data = dataset_[dataset_.columns[~dataset_.columns.isin(cat_cols)]]
    objective_var = numeric_data[['causa_retiro']]
    numeric_data = numeric_data.drop('causa_retiro', axis=1)
    #setting dtypes
    numeric_data = numeric_data.astype({'anios': int})
    dummies = (pd.get_dummies(cat_dataset)*1).drop('genero_F', axis=1)
    dataset_ = dummies.join(numeric_data).join(objective_var)
    #encoding scope variable
    if labeling_scope:
        dataset_.loc[dataset_.causa_retiro=='TERMINACION DE CONTRATO', 'causa_retiro'] = 1
        dataset_.loc[dataset_.causa_retiro!=1, 'causa_retiro'] = 0
    return dataset_