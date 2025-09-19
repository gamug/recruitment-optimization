import json, os, datetime
from typing import Tuple, Any
import pandas as pd, numpy as np
from scipy import stats
from src.commons.tools import input_path, output_path, check_directories, input_numeric_col

pd.set_option("display.max_columns", None)

check_directories()

cat_cols = ['Desc_Cargo', 'Proyecto', 'genero']


def feature_dane(df: pd.DataFrame):
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
    featured_dataset = featured_dataset.drop(np.unique(drop_vars), axis=1)
    return featured_dataset

def get_high_correlated_features(df: pd.DataFrame) -> pd.DataFrame:
    corr_matrix = df.corr().abs()
    corr_matrix = corr_matrix[(corr_matrix>0.8)&(corr_matrix<1)]
    filter_ = corr_matrix.isna().all()
    hyper_correlated = corr_matrix[corr_matrix.columns[~corr_matrix.columns.isin(filter_[filter_].index.tolist())]]
    v1, v2, corr = [], [], []
    for col in hyper_correlated.columns:
        for idx in hyper_correlated.index:
            if not pd.isna(hyper_correlated[col].loc[idx]):
                aux = [col, idx]
                aux.sort()
                v1.append(aux[0])
                v2.append(aux[1])
                corr.append(hyper_correlated[col].loc[idx])
    hyper_correlated = pd.DataFrame({'variable1': v1, 'variable2': v2, 'correlation': corr}).drop_duplicates()
    return hyper_correlated

def years_computing(file_path: str) -> pd.DataFrame:
    desc_cargo_eq = {
        "CONDUCTOR VOLQUETA DAF": "CONDUCTOR DE VOLQUETA DAF",
        "AUXILIAR ADMINISTRATIVA": "AUXILIAR ADMINSTRATIVO",
        "INSPECTOR SST": "INSPECTOR SST I",
        "SOLDADOR ": "SOLDADOR I"
    }
    dataset = pd.read_csv(
        file_path,
        parse_dates=['fecha_nacimiento']
    )
    dataset['Desc_Cargo'] = dataset['Desc_Cargo'].replace(desc_cargo_eq)
    dataset_ = dataset.copy()
    dataset_.insert(10, 'anios', (datetime.datetime.now()-dataset.fecha_nacimiento).dt.days//365.25)
    dataset_ = dataset_.drop('fecha_nacimiento', axis=1)
    dataset_ = dataset_[~(dataset_.causa_retiro=='MUERTE DEL TRABAJADOR')]
    return dataset_

def outliers_remotion(dataset_: pd.DataFrame) -> pd.DataFrame:
    dataset_.loc[dataset_.anios<18, 'anios'] = np.nan
    dataset_.loc[dataset_.anios>60, 'anios'] = np.nan
    dataset_ = input_numeric_col(dataset_, 'anios')
    return dataset_

def get_dummies(dataset_: pd.DataFrame) -> pd.DataFrame:
    cat_dataset = dataset_[cat_cols]
    numeric_data = dataset_[dataset_.columns[~dataset_.columns.isin(cat_cols)]]
    objective_var = numeric_data[['causa_retiro']]
    numeric_data = numeric_data.drop('causa_retiro', axis=1)
    #setting dtypes
    numeric_data = numeric_data.astype({'anios': int})
    dummies = (pd.get_dummies(cat_dataset)*1).drop('genero_F', axis=1)
    dataset_ = dummies.join(numeric_data).join(objective_var)
    #encoding scope variable
    dataset_.loc[dataset_.causa_retiro=='TERMINACION DE CONTRATO', 'causa_retiro'] = 1
    dataset_.loc[dataset_.causa_retiro!=1, 'causa_retiro'] = 0
    return dataset_

def drop_non_variant_cols(dataset_: pd.DataFrame) -> pd.DataFrame:
    #Droping columns with unique values
    no_variation_cols = dataset_.corr().isna().all()
    no_variation_cols = no_variation_cols[no_variation_cols].index.tolist()
    dataset_ = dataset_[dataset_.columns[~dataset_.columns.isin(no_variation_cols)]]
    return dataset_

def droping_irrelevant_variables(dataset_: pd.DataFrame, file_path: str) -> pd.DataFrame:
    #Droping columns with no correlation with objective variable ('causa_retiro')
    corr_matrix = dataset_.corr()
    corr_matrix.to_excel(os.path.join(os.path.dirname(file_path), 'correlation_matrix.xlsx'), index=0)
    relevant_variables = corr_matrix.loc['causa_retiro'].abs()
    relevant_variables = relevant_variables[relevant_variables>0.05].index.tolist()
    dataset_ = dataset_[relevant_variables]
    return dataset_

def droping_redundant_variables(dataset_: pd.DataFrame) -> pd.DataFrame:
    #Droping highly correlated columns
    print('         computing dane features...')
    featured_dataset = feature_dane(dataset_)
    correlated_features = get_high_correlated_features(featured_dataset) #use this to get highly correlated variables
    cols_high_correlated = [
        'Desc_Cargo_AYUDANTE DE OBRA tasa 6.96', 'CTNENCUEST', 'TP16_HOG',
        'TVIVIENDA', 'TP9_2_2_MI', 'TP19_RECB1', 'TP19_INTE2', 'TP19_EE_1',
        'TP19_ALC_1', 'TP19_INTE9', 'TP15_4_OCU', 'TP19_RECB2'
    ]
    featured_dataset = featured_dataset.drop(cols_high_correlated, axis=1)
    scope = featured_dataset.causa_retiro
    featured_dataset = featured_dataset.drop('causa_retiro', axis=1)
    featured_dataset['retiro'] = scope
    return featured_dataset

def process_deploy_set(dataset_: pd.DataFrame) -> pd.DataFrame:
    print('         computing dane features...')
    featured_dataset = feature_dane(dataset_)
    featured_dataset = featured_dataset.drop('causa_retiro', axis=1)
    featured_dataset['retiro'] = '?'
    schema = os.path.join('..', 'input', 'prediction-data-mining-schema.json')
    with open(schema) as f:
        schema = json.loads(f.read())
    missing_cols = list(set(schema['schema'])-set(featured_dataset.columns))
    for col in missing_cols:
        featured_dataset[col] = 0
    featured_dataset = featured_dataset[schema['schema']]
    return featured_dataset

def process_prediction_dataset(file_path: str) -> pd.DataFrame:
    set_ = os.path.basename(file_path).split('_')[0]
    print(f'getting {set_} dataset...')
    print('     computing features...')
    dataset_ = years_computing(file_path)
    print('     removing outliers...')
    dataset_ = outliers_remotion(dataset_)
    print('     getting dummies...')
    dataset_ = get_dummies(dataset_)
    if set_=='train':
        print('     dropping unvariant cols...')
        dataset_ = drop_non_variant_cols(dataset_)
        print('     dropping irrelevant variables...')
        dataset_ = droping_irrelevant_variables(dataset_, file_path)
        print('     dropping redundant variables...')
        featured_dataset = droping_redundant_variables(dataset_)
    else:
        print('     processing deploy dataset...')
        featured_dataset = process_deploy_set(dataset_)
    print(f'    saving {set_} dataset')
    featured_dataset.to_csv(os.path.join(os.path.dirname(file_path), 'non_correlated_dataset.csv'), index=0)
    return featured_dataset

def get_train_deploy_datasets():
    file_path = os.path.join(output_path, 'predictive_mining', 'train_set', 'train_without_featuring.csv')
    process_prediction_dataset(file_path)
    file_path = os.path.join(output_path, 'predictive_mining', 'deploy_set', 'deploy_without_featuring.csv')
    process_prediction_dataset(file_path)
    
if __name__=='__main__':
    get_train_deploy_datasets()