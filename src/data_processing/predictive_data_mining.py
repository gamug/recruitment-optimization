import json, os
import pandas as pd, numpy as np
from scipy import stats
import src.commons.tools as data_tools

pd.set_option("display.max_columns", None)

cat_cols = ['Desc_Cargo', 'Proyecto', 'genero']

def read_data(file_path: str) -> pd.DataFrame:
    '''Read dataset from the specified file path.
    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the curated dataset without feature engineering.}
    Returns
    -------
    pd.DataFrame
        The loaded dataset as a pandas DataFrame.'''
    dataset = pd.read_csv(
        file_path,
        parse_dates=['fecha_nacimiento']
    )
    return dataset

def get_high_correlated_features(df: pd.DataFrame) -> pd.DataFrame:
    ''' Identify highly correlated features in the DataFrame.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame coming from dane feature engineering to analyze for high correlations.
    Returns
    -------
    pd.DataFrame
        DataFrame containing pairs of highly correlated features and their correlation values.'''
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

def drop_non_variant_cols(dataset_: pd.DataFrame) -> pd.DataFrame:
    '''Drop columns with no variation in the dataset.
    Parameters
    ----------
    dataset_ : pd.DataFrame
        The dataset with feature engineering, outliers remotion and dummies creation,
        from which to drop non-variant columns.
    Returns
    -------
    pd.DataFrame
        The dataset with non-variant columns removed.'''
    #Droping columns with unique values
    no_variation_cols = dataset_.corr().isna().all()
    no_variation_cols = no_variation_cols[no_variation_cols].index.tolist()
    dataset_ = dataset_[dataset_.columns[~dataset_.columns.isin(no_variation_cols)]]
    return dataset_

def dropping_irrelevant_variables(dataset_: pd.DataFrame, file_path: str) -> pd.DataFrame:
    '''Drop irrelevant predictors from the dataset based on correlation with the target variable.
    Parameters
    ----------
    dataset_ : pd.DataFrame
        The dataset with feature engineering, outliers remotion and dummies creation,
        from which to drop irrelevant predictors.
    file_path : str
        The path to the CSV file containing the curated dataset without feature engineering,
        used to determine the location for saving the correlation matrix.'''
    #Droping columns with no correlation with objective variable ('causa_retiro')
    corr_matrix = dataset_.corr()
    corr_matrix.to_excel(os.path.join(os.path.dirname(file_path), 'correlation_matrix.xlsx'), index=0)
    relevant_variables = corr_matrix.loc['causa_retiro'].abs()
    relevant_variables = relevant_variables[relevant_variables>0.05].index.tolist()
    dataset_ = dataset_[relevant_variables]
    return dataset_

def dropping_redundant_variables(dataset_: pd.DataFrame) -> pd.DataFrame:
    '''Drop highly correlated predictors from the dataset.
    Parameters
    ----------
    dataset_ : pd.DataFrame
        The dataset with feature engineering, outliers remotion and dummies creation,
        from which to drop highly correlated predictors.
    Returns
    -------
    pd.DataFrame
        The dataset with highly correlated predictors removed.'''
    #Droping highly correlated columns
    print('         computing dane features...')
    featured_dataset = data_tools.feature_dane(dataset_)
    correlated_features = get_high_correlated_features(featured_dataset) #use this to get highly correlated variables
    featured_dataset = featured_dataset.drop(data_tools.cols_high_correlated, axis=1)
    scope = featured_dataset.causa_retiro
    featured_dataset = featured_dataset.drop('causa_retiro', axis=1)
    featured_dataset['retiro'] = scope
    return featured_dataset

def process_deploy_set(dataset_: pd.DataFrame) -> pd.DataFrame:
    '''Process the deploy dataset by computing features and aligning it with the training dataset schema.
    Parameters
    ----------
    dataset_ : pd.DataFrame
        The deploy dataset with feature engineering, outliers remotion and dummies creation.
    Returns
    -------
    pd.DataFrame
        The processed deploy dataset aligned with the training dataset schema.'''
    print('         computing dane features...')
    featured_dataset = data_tools.feature_dane(dataset_)
    featured_dataset = featured_dataset.drop('causa_retiro', axis=1)
    featured_dataset['retiro'] = '?'
    schema = os.path.join('..', 'input', 'data-mining-schema.json')
    with open(schema) as f:
        schema = json.loads(f.read())
    missing_cols = list(set(schema['schema'])-set(featured_dataset.columns))
    for col in missing_cols:
        featured_dataset[col] = 0
    featured_dataset = featured_dataset[schema['schema']]
    return featured_dataset

def process_prediction_dataset(file_path: str, prefix: str='') -> pd.DataFrame:
    '''Process the prediction dataset by reading, computing features, removing outliers,
    getting dummies, and saving the processed dataset.
    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the curated dataset without feature engineering.
    prefix : str, optional
        Prefix to identify the files input-output. Like an unique identifier to make a trace between tests, by default ''
    Returns
    -------
    pd.DataFrame
        The processed prediction dataset.'''
    set_ = os.path.basename(file_path).split('_')[1]
    print(f'getting {set_} dataset...')
    dataset = read_data(file_path)
    print('     computing features...')
    dataset_ = data_tools.years_computing(dataset)
    print('     removing outliers...')
    dataset_ = data_tools.outliers_remotion(dataset_)
    print('     getting dummies...')
    dataset_ = data_tools.get_dummies(dataset_, cat_cols)
    if set_=='train':
        print('     dropping unvariant cols...')
        dataset_ = drop_non_variant_cols(dataset_)
        print('     dropping irrelevant variables...')
        dataset_ = dropping_irrelevant_variables(dataset_, file_path)
        print('     dropping redundant variables...')
        featured_dataset = dropping_redundant_variables(dataset_)
    else:
        print('     processing deploy dataset...')
        featured_dataset = process_deploy_set(dataset_)
    print(f'    saving {set_} dataset')
    featured_dataset.to_csv(os.path.join(os.path.dirname(file_path), f'{prefix}_non_correlated_dataset_{set_}.csv'), index=0)
    return featured_dataset

def get_train_deploy_datasets(prefix: str=''):
    '''Generate and save the train and deploy datasets by processing the respective CSV files.
    Parameters
    ----------
    prefix : str, optional
        Prefix to identify the files input-output. Like an unique identifier to make a trace between tests, by default ''
    Returns
    -------
    None
        The function saves the processed train and deploy datasets to CSV files and does not return any value.'''
    file_path = os.path.join(data_tools.output_path, 'predictive_mining', 'train_set', f'{prefix}_train_without_featuring.csv')
    process_prediction_dataset(file_path, prefix)
    file_path = os.path.join(data_tools.output_path, 'predictive_mining', 'deploy_set', f'{prefix}_deploy_without_featuring.csv')
    process_prediction_dataset(file_path, prefix)
    
if __name__=='__main__':
    get_train_deploy_datasets()