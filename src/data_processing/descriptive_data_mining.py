import os
from typing import Tuple
import pandas as pd, numpy as np
import src.commons.tools as data_tools

cat_cols = ['Desc_Cargo', 'Proyecto', 'genero', 'id_tipo_contrato', 'id_estado_civil', 'id_turno', 'NMB_LC_CM']

def read_data(prefix: str='') -> Tuple[pd.DataFrame]:
    '''reads the enriched dane database and the dictionaries of variables
    Parameters
    ----------
    prefix : str, optional
        Prefix to identify the files input-output. Like an unique identifier to make a trace between tests, by default ''
    Returns
    -------
    Tuple[pd.DataFrame]
        Tuple with the DataFrames: dane_enriched, dane_dict, business_dict'''
    training_set = os.path.join(data_tools.output_path, 'databases', f'{prefix}_dane_enriched_db.csv')
    dane_enriched = pd.read_csv(
        training_set,
        parse_dates=['fecha_ingreso', 'fecha_final', 'fecha_retiro', 'fecha_nacimiento']
    )
    dane_dict = pd.read_excel(
        os.path.join(data_tools.input_path, 'DICCIONARIO_DATOS_DANE.xlsx'),
        sheet_name='MGN_ANM_MANZANA',
        skiprows=6
    )
    business_dict = pd.read_excel(
        os.path.join(data_tools.input_path, 'DICCIONARIO 1.xlsx'),
        sheet_name='DICCIONARIO FINAL',
        skiprows=3
    ).drop('Unnamed: 0', axis=1)
    return dane_enriched, dane_dict, business_dict

def read_data(file_path: str) -> pd.DataFrame:
    dataset = pd.read_csv(
            file_path,
            parse_dates=['fecha_nacimiento', 'fecha_ingreso', 'fecha_retiro']
        )
    return dataset

def drop_unvariant_cols(dataset: pd.DataFrame) -> pd.DataFrame:
    '''Drop unvariant columns in the DataFrame
    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame to process
    Returns
    -------
    pd.DataFrame
        DataFrame without unvariant columns'''
    cat_vars = cat_cols.copy()
    cat_vars.append('causa_retiro')
    dataset_cats = dataset[cat_vars]
    dataset_num = dataset.drop(cat_vars, axis=1)
    corr = dataset_num.corr()
    drop_cols = corr[corr.isna().all()].index
    dataset_num = dataset_num.drop(drop_cols, axis=1)
    dataset_ = dataset_cats.join(dataset_num)
    return dataset_

def descriptive_base_processing(prefix: str=''):
    '''Process the descriptive base data.
    Parameters
    ----------
    prefix : str, optional
        Prefix to identify the files input-output. Like an unique identifier to make a trace between tests, by default ''
    Returns
    -------
    pd.DataFrame
        DataFrame processed and ready to use in descriptive modeling'''
    dataset = read_data(prefix)
    dataset_ = data_tools.years_computing(dataset)
    #computing permanence contract time
    dataset_['permanencia'] = (dataset_['fecha_retiro']-dataset_['fecha_ingreso']).dt.days.astype(int)
    dataset_.loc[dataset_.permanencia<=0, 'permanencia'] = np.nan
    dataset_ = data_tools.input_numeric_col(dataset_, 'permanencia').drop(['fecha_ingreso', 'fecha_retiro'], axis=1)
    #removing outliers
    print('     removing outliers...')
    dataset_ = data_tools.outliers_remotion(dataset_)
    cat_vars = cat_cols.copy()
    cat_vars.append('causa_retiro')
    dataset_cats = dataset_[cat_vars]
    dataset_num = dataset_.drop(cat_vars, axis=1)
    dataset_num = data_tools.feature_dane(dataset_num)
    dataset_ = dataset_num.join(dataset_cats)
    dataset_ = drop_unvariant_cols(dataset_)
    return dataset_

def numeric_binner(dataset: pd.DataFrame) -> pd.DataFrame:
    '''Generate a categorical DataFrame by binning numeric variables into quartiles.
    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame to process
    Returns
    -------
    pd.DataFrame
        Categorical DataFrame with binned numeric variables'''
    cat_vars = cat_cols.copy()
    cat_vars.append('causa_retiro')
    dataset_cats = dataset[cat_vars]
    dataset_num = dataset.drop(cat_vars, axis=1)
    quartils = [0, .25, .5, .75, 1.]
    dataset_num = dataset_num.apply(lambda col: pd.qcut(col, q=quartils, duplicates="drop"))
    dataset_ = dataset_num.join(dataset_cats)
    return dataset_

def save_data(dataset_cluster: pd.DataFrame, categorical_db: pd.DataFrame, file_path: str, prefix: str='') -> None:
    dataset_cluster.to_csv(
        os.path.join(os.path.dirname(file_path), f'{prefix}_description_numeric.csv'),
        index=0
    )
    categorical_db.to_csv(
        os.path.join(os.path.dirname(file_path), f'{prefix}_description_categorical.csv'),
        index=0
    )

def process_descriptive_sets(prefix: str='') -> None:
    '''Process and save the descriptive datasets.
    Parameters
    ----------
    prefix : str, optional
        Prefix to identify the files input-output. Like an unique identifier to make a trace between tests, by default ''
    returns
    -------
    None
        Saves the processed datasets to CSV files'''
    print('processing descriptive sets...')
    file_path = os.path.join(data_tools.output_path, 'descriptive_mining', f'{prefix}_descriptive_without_featuring.csv')
    dataset = descriptive_base_processing(file_path)
    print('     getting dummies...')
    dataset_cluster = data_tools.get_dummies(dataset, cat_cols)
    dataset_cluster = dataset_cluster.drop(data_tools.cols_high_correlated, axis=1)
    print('     generating categorical db (quantils)...')
    categorical_db = numeric_binner(dataset)
    categorical_db = categorical_db.drop(data_tools.cols_high_correlated[1:], axis=1)
    print('     saving datasets...')
    save_data(dataset_cluster, categorical_db, file_path, prefix=prefix)

if __name__=='__main__':
    process_descriptive_sets()