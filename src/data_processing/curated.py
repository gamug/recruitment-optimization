import os
from typing import Tuple
import pandas as pd
import src.commons.tools as data_tools


def read_data(prefix: str='') -> Tuple[pd.DataFrame]:
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

def build_sets(base_curated: pd.DataFrame, prefix: str) -> None:
    train_set = base_curated[base_curated.causa_retiro!='Activo'].drop(
        ['fecha_retiro', 'fecha_final', 'fecha_ingreso'],
        axis=1
    )
    test_set = base_curated[base_curated.causa_retiro=='Activo'].drop(
        ['fecha_retiro', 'fecha_final', 'fecha_ingreso'],
        axis=1
    )
    descriptive = base_curated[base_curated.causa_retiro!='Activo']
    train_set.to_csv(os.path.join(data_tools.output_path, 'predictive_mining', 'train_set', f'{prefix}_train_without_featuring.csv'), sep=',', index=0)
    test_set.to_csv(os.path.join(data_tools.output_path, 'predictive_mining', 'deploy_set', f'{prefix}_deploy_without_featuring.csv'), sep=',', index=0)
    descriptive.to_csv(os.path.join(data_tools.output_path, 'descriptive_mining', f'{prefix}_descriptive_without_featuring.csv'), sep=',', index=0)

def curate_without_featuring(prefix: str=''):
    print('process curated data...')
    print('     reading inputs...')
    dane_enriched, dane_dict, business_dict = read_data(prefix)
    print('     input missing values...')
    base_curated, _ = data_tools.input_missing_values(dane_enriched, dane_dict, business_dict)
    print('     saving sets...')
    build_sets(base_curated, prefix=prefix)
    
if __name__=='__main__':
    curate_without_featuring()