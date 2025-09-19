import json, os
from typing import Tuple, Any
import pandas as pd
from scipy import stats
from src.commons.tools import input_path, output_path, input_numeric_col


training_set = os.path.join(output_path, 'databases', 'dane_enriched_db.csv')

with open(os.path.join(input_path, 'column-curated.json'), 'r', encoding='utf-8') as f:
    column_drops = json.loads(f.read())

def read_data(enriched_data: str=training_set) -> Tuple[pd.DataFrame]:
    dane_enriched = pd.read_csv(
        enriched_data,
        parse_dates=['fecha_ingreso', 'fecha_final', 'fecha_retiro', 'fecha_nacimiento']
    )
    dane_dict = pd.read_excel(
        os.path.join(input_path, 'DICCIONARIO_DATOS_DANE.xlsx'),
        sheet_name='MGN_ANM_MANZANA',
        skiprows=6
    )
    business_dict = pd.read_excel(
        os.path.join(input_path, 'DICCIONARIO 1.xlsx'),
        sheet_name='DICCIONARIO FINAL',
        skiprows=3
    ).drop('Unnamed: 0', axis=1)
    return dane_enriched, dane_dict, business_dict

def input_missing_values(
    dane_enriched: pd.DataFrame,
    dane_dict: pd.DataFrame,
    business_dict: pd.DataFrame
    ) -> Tuple[Any]:
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
                    input_numeric_col(base_curated, col)
            else:
                dropped_cols.append(col)
                base_curated = base_curated.drop(col, axis=1)
    return base_curated, dropped_cols

def build_sets(base_curated: pd.DataFrame) -> None:
    train_set = base_curated[base_curated.causa_retiro!='Activo'].drop(
        ['fecha_retiro', 'fecha_final', 'fecha_ingreso'],
        axis=1
    )
    test_set = base_curated[base_curated.causa_retiro=='Activo'].drop(
        ['fecha_retiro', 'fecha_final', 'fecha_ingreso'],
        axis=1
    )
    descriptive = base_curated[base_curated.causa_retiro!='Activo']
    train_set.to_csv(os.path.join(output_path, 'predictive_mining', 'train_set', 'train_without_featuring.csv'), sep=',', index=0)
    test_set.to_csv(os.path.join(output_path, 'predictive_mining', 'deploy_set', 'deploy_without_featuring.csv'), sep=',', index=0)
    descriptive.to_csv(os.path.join(output_path, 'descriptive_mining', 'descriptive_without_featuring.csv'), sep=',', index=0)

def curate_without_featuring():
    print('process curated data...')
    print('     reading inputs...')
    dane_enriched, dane_dict, business_dict = read_data()
    print('     input missing values...')
    base_curated, _ = input_missing_values(dane_enriched, dane_dict, business_dict)
    print('     saving sets...')
    build_sets(base_curated)
    
if __name__=='__main__':
    curate_without_featuring()