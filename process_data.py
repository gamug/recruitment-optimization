from src.data_processing.precurated import preprocess_data
from src.data_processing.geocode_data import geocoding
from src.data_processing.curated import curate_without_featuring
from src.data_processing.predictive_data_mining import get_train_deploy_datasets
from src.commons.tools import check_directories

check_directories()

if __name__=='__main__':
    prefix = 'test'
    preprocess_data(prefix=prefix)
    geocoding(geocode_data=False, merge_dane=False, prefix=prefix)
    curate_without_featuring(prefix=prefix)
    get_train_deploy_datasets(prefix=prefix)