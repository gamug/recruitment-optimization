import os, requests, time, tqdm
from dotenv import load_dotenv
import pandas as pd, geopandas as gpd, numpy as np

from src.commons.tools import input_path, output_path

load_dotenv()

location_drops = [
    'id_ciudad', 'id_area', 'direccion', 'id_ciudad_residencia', 'barrio',
    'id_departamento_exp', 'id_departamento_res'
]
here_api_key = os.environ.get('HERE_API_KEY')

def extract_data(result: dict, cities: list, districts: list, latitudes: list, longitudes: list) -> None:
    ''' Extract relevant data from geocoding result. This receives the Here API response
        as a dictionary and appends the relevant data to the lists provided as arguments.
    Parameters
    ----------
    result : dict
        The response from the Here API geocoding request.
    cities : list
        List to append the extracted city names.
    districts : list
        List to append the extracted district names.
    latitudes : list
        List to append the extracted latitude values.
    longitudes : list
        List to append the extracted longitude values.
    Returns
    -------
    None
        The function modifies the input lists in place and does not return any value.'''
    if not 'error' in result and len(result['items']):
        if 'city' in result['items'][0]['address']:
            cities.append(result['items'][0]['address']['city'])
        else:
            cities.append(None)
        if 'district' in result['items'][0]['address']:
            districts.append(result['items'][0]['address']['district'])
        else:
            districts.append(None)
        latitudes.append(result['items'][0]['position']['lat'])
        longitudes.append(result['items'][0]['position']['lng'])
    elif not 'error' in result and not len(result['items']):
        cities.append(None)
        districts.append(None)
        latitudes.append(None)
        longitudes.append(None)
        
def geocode_precurated(precurated, prefix: str='') -> pd.DataFrame:
    ''' Geocode the addresses in the precurated dataframe using the Here API.
    Parameters
    ----------
    precurated : pd.DataFrame
        DataFrame containing the precurated data with address information.
    prefix : str, optional
        Prefix to identify the files input-output. Like an unique identifier to make a trace between tests, by default ''
    Returns
    -------
    pd.DataFrame
        DataFrame with geocoded information including city, district, latitude, and longitude.'''
    department, city = 'Cundinamarca', 'Bogotá'
    cities, districts, latitudes, longitudes = [], [], [], []

    base_url = 'https://geocode.search.hereapi.com/v1/geocode?limit=2&q={full_address}&apiKey={here_api_key}'

    df, k_limit = precurated, 1
    pbar = tqdm.tqdm(range(len(df)))
    for i in pbar:
        address, district = df.direccion.iloc[i], df.barrio.iloc[i]
        if pd.isna(address) or pd.isnull(address):
            full_address = None
        else:
            if pd.isna(district) or pd.isnull(district):
                full_address = f'{address}, {city}, {department}'
            else:
                full_address = f'{address}, {district}, {city}, {department}'
        query = base_url.format(full_address=full_address, here_api_key=here_api_key).replace(' ', '%20')
        if not full_address is None:
            result = requests.get(query).json()
            extract_data(result, cities, districts, latitudes, longitudes)
            k = 0
            while 'error' in result and k<k_limit:
                t = k*10
                time.sleep(t)
                result = requests.get(query).json()
                extract_data(result, cities, districts, latitudes, longitudes)
                k += 1
            if 'error' in result and k==k_limit:
                cities.append(None)
                districts.append(None)
                latitudes.append(None)
                longitudes.append(None)
        else:
            cities.append(None)
            districts.append(None)
            latitudes.append(None)
            longitudes.append(None)
    geocoded = df.\
        assign(city=cities).\
            assign(district=districts).\
                assign(latitude=latitudes).\
                    assign(longitude=longitudes)
    geocoded = geocoded.drop(location_drops, axis=1)
    geocoded.to_csv(
            os.path.join(output_path, 'databases', f'{prefix}_geocoded.csv'),
            index=0,
            sep=','
        )
    return geocoded

def enrich_with_dane():
    ''' Enrich the geocoded data with DANE microdata by performing a spatial join.
    Returns
    -------
    pd.DataFrame
        DataFrame enriched with DANE microdata.'''
    geocoded = pd.read_csv(os.path.join(output_path, 'databases', 'geocoded.csv'))
    geocoded = geocoded[~geocoded.latitude.isnull()]
    # Convert geocode into GeoDataFrame with geometry from lat/lon
    print('         charging DANE data...')
    dane = gpd.read_file(f'zip://{input_path}/DANE_microdata_2018.zip')
    dane = dane.to_crs(epsg=3857)
    print('         joining dataframes with dinamic spacing...')
    dfs = []
    pbar = tqdm.tqdm(geocoded.iterrows(), total=geocoded.shape[0])
    for _, row in pbar:
        value, k = np.nan, 1
        while pd.isna(value):
            result = row.to_frame().T
            result = gpd.GeoDataFrame(
                result,
                geometry=gpd.points_from_xy(result['longitude'], result['latitude']),
                crs="EPSG:4326"  # WGS84 lat/lon
            ).to_crs(epsg=3857)
            result = gpd.sjoin_nearest(result, dane, how="left", max_distance=0.01*k)
            if not pd.isna(result.COD_DANE_A.iloc[0]):
                dfs.append(result)
            else:
                k *= 10
            value = result.COD_DANE_A.iloc[0]
    geocoded_dane = pd.concat(dfs)
    return geocoded_dane

def save_results(df: pd.DataFrame, prefix: str=''):
    ''' Save the enriched DataFrame to a CSV file.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be saved.
    prefix : str, optional
        Prefix to identify the files input-output. Like an unique identifier to make a trace between tests, by default ''
    Returns
    -------
    None
        The function saves the DataFrame to a CSV file and does not return any value.'''
    df.to_csv(
        os.path.join(output_path, 'databases', f'{prefix}_dane_enriched_db.csv'),
        index=0,
        sep=','
    )

def geocoding(geocode_data=False, merge_dane=False, prefix: str='') -> pd.DataFrame:
    ''' Geocode and optionally enrich the precurated data with DANE microdata.
    Parameters
    ----------
    geocode_data : bool, optional
        Whether to perform geocoding of the precurated data, by default False
    merge_dane : bool, optional
        Whether to merge the geocoded data with DANE microdata, by default False
    prefix : str, optional
        Prefix to identify the files input-output. Like an unique identifier to make a trace between tests, by default ''
    Returns
    -------
    pd.DataFrame
        DataFrame containing the geocoded and optionally DANE-enriched data.'''
    data = {}
    print('geocoding data...')
    precurated = pd.read_csv(os.path.join(output_path, 'databases', f'{prefix}_precurated.csv'))
    if geocode_data:
        print('     gecoding precurated data...')
        data['geocoded'] = geocode_precurated(precurated)
    if merge_dane:
        print('     merging precurated geocoded data with DANE...')
        df = enrich_with_dane()
        print('     saving data...')
        save_results(df, prefix)
    data['geocoded_dane'] = pd.read_csv(os.path.join(output_path, 'databases', f'{prefix}_dane_enriched_db.csv'))
    return data

if __name__=='__main__':
    geocoding()