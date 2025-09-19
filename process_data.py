from src.data_processing.precurated import preprocess_data
from src.data_processing.geocode_data import geocoding
from src.data_processing.curated import curate_without_featuring
from src.data_processing.predictive_data_mining import get_train_deploy_datasets
from src.commons.tools import check_directories

check_directories()

if __name__=='__main__':
    preprocess_data()
    geocoding(geocode_data=False, merge_dane=False)
    curate_without_featuring()
    get_train_deploy_datasets()
    