import json, os
from typing import Tuple, Any
import pandas as pd
import src.commons.tools as data_tools


with open(os.path.join(data_tools.input_path, 'column-curated.json'), 'r', encoding='utf-8') as f:
    column_drops = json.loads(f.read())

def read_data(prefix: str='') -> Tuple[pd.DataFrame]:
    '''Read input data
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

def input_missing_values(
    dane_enriched: pd.DataFrame,
    dane_dict: pd.DataFrame,
    business_dict: pd.DataFrame
    ) -> Tuple[Any]:
    '''Input missing values in the DataFrame
    Parameters
    ----------
    dane_enriched : pd.DataFrame
        DataFrame with missing values. The DataFrame is the one coming from geocode_data script
    dane_dict : pd.DataFrame
        DataFrame with the DANE data dictionary
    business_dict : pd.DataFrame
        DataFrame with the business data dictionary
    Returns
    -------
    Tuple[Any]
        Tuple with the DataFrame with inputed missing values and a list with the dropped columns'''
    base_curated = dane_enriched.drop(column_drops['irrelevant_cols'], axis=1)
    base_curated = base_curated.drop(column_drops['geocoded_dane_col_drops'], axis=1)
    null_counts = pd.DataFrame({col: [round(base_curated[col].isna().sum()*100/len(base_curated), 2)] for col in base_curated.columns}).T
    dropped_cols = []
    for col in null_counts.index:
        if null_counts.loc[col].iloc[0]:
            if null_counts.loc[col].iloc[0]<=15:
                if col in dane_dict.VARIABLE.tolist():
                    discrete = dane_dict[dane_dict.VARIABLE==col].TIPO.iloc[0] in ['Text', 'Long Integer']
                else:
                    discrete =  business_dict[business_dict['Variable']==col].iloc[0]=='Discreta'
                if discrete:
                    base_curated[col] = base_curated[col].fillna(base_curated[col].mode().iloc[0])
                else:
                    base_curated = data_tools.input_numeric_col(base_curated, col)
            else:
                dropped_cols.append(col)
                base_curated = base_curated.drop(col, axis=1)
    return base_curated, dropped_cols

def build_sets(base_curated: pd.DataFrame, prefix: str) -> None:
    '''Build the sets for predictive and descriptive mining
    Parameters
    ----------
    base_curated : pd.DataFrame
        DataFrame with the curated data
    prefix : str
        Prefix to identify the files input-output. Like an unique identifier to make a trace between tests
    Returns
    -------
    None'''
    train_set = base_curated[base_curated.causa_retiro!='Activo'].drop(
        ['fecha_retiro', 'fecha_final', 'fecha_ingreso', 'NMB_LC_CM'],
        axis=1
    )
    test_set = base_curated[base_curated.causa_retiro=='Activo'].drop(
        ['fecha_retiro', 'fecha_final', 'fecha_ingreso', 'NMB_LC_CM'],
        axis=1
    )
    descriptive = base_curated[base_curated.causa_retiro!='Activo'].drop(
        ['fecha_final', 'id_destino', 'id_nivel_academico', 'subsidio_tte'],
        axis=1
    )
    train_set.to_csv(os.path.join(data_tools.output_path, 'predictive_mining', 'train_set', f'{prefix}_train_without_featuring.csv'), sep=',', index=0)
    test_set.to_csv(os.path.join(data_tools.output_path, 'predictive_mining', 'deploy_set', f'{prefix}_deploy_without_featuring.csv'), sep=',', index=0)
    descriptive.to_csv(os.path.join(data_tools.output_path, 'descriptive_mining', f'{prefix}_descriptive_without_featuring.csv'), sep=',', index=0)

def curate_without_featuring(prefix: str=''):
    '''Curate data without featuring
    Parameters
    ----------
    prefix : str, optional
        Prefix to identify the files input-output. Like an unique identifier to make a trace between tests, by default ''
    Returns
    -------
    None'''
    print('process curated data...')
    print('     reading inputs...')
    dane_enriched, dane_dict, business_dict = read_data(prefix)
    print('     input missing values...')
    base_curated, _ = input_missing_values(dane_enriched, dane_dict, business_dict)
    print('     saving sets...')
    build_sets(base_curated, prefix=prefix)
    
if __name__=='__main__':
    curate_without_featuring()