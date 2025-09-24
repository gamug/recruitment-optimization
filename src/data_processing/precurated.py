#ALGUNOS COMENTARIOS DE COLUMNAS
#sanguineo, rh más del 60% de los datos son nulos
#dias_compensatorios_pendientes sólo dos registros son diferentes de 0
#forma_pago 8950 registros son B y el resto E (menos de 30)
#id_tipo_empleado solo tiene 7 valores diferentes de 1
#id_sub_tipo_cotizante tiene casi todos sus datos en 0 sin guardar relación con nada aparente en sus grupos
#peso y estatura se eliminan debido a que mas de 8000 valores vienen en 0

#DEFINICIÓN DE LINEA BASE:
# ### Consideraciones:
# - Sólo se tuvieron en cuenta los registros con 'Planta'='OPERARIOS'.
# - Sólo se tuvieron en cuenta los registros con 'Proyecto' en ['Contrato Subestructura S2', 'Contrato Puente calle 26', 'WF4'].
# - Se eliminaron las columnas irrelevantes, inútiles y duplicadas.
# - Se convirtieron las columnas de fechas a formato datetime.
# - Se renombraron las columnas 'id.1', 'fecha_ingreso.1' y 'fecha_retiro.1' a 'id_unico_empleado', 'fecha_ingreso' y 'fecha_retiro' respectivamente
# - Algunos registros tienen 'fecha_retiro.1'='1/01/1900', estos registros fueron retirados asumiendo que son empleados que continuan en la compañía.
# - Se considera que el contrato mínimo es a 3 meses (120 días). Se filtra este tiempo de permanencia para hallar el caso base.
# - Se contruye un histograma para observar la distribución de los tiempos de permanencia, estos se ven como una distribución unifome con un leve sesgoi hacia la baja.
# - Se halla la desviación estándar y se observa que es alta, por lo que se decide tomar como métrica para la línea base.
# - La métrica de la línea base es la desviación estándar del tiempo de permanencia en días, considerando los puntos antes mencionados.

import json, os
import pandas as pd
import matplotlib.pyplot as plt

from src.commons.tools import input_path, output_path

plt.style.use('seaborn-v0_8')
pd.set_option('display.max_columns', None)


with open(os.path.join(input_path, 'colum-cleaning.json'), 'r', encoding='utf-8') as f:
    cols = json.loads(f.read())

def read_inputs() -> dict:
    ''' Read input data files and return them in a dictionary.
    Returns
    -------
    dict
        A dictionary containing the input dataframes.
        employees_: DataFrame containing employee data from 'Empleados_AR.csv'.'
        drops: DataFrame containing drop data from 'Retiros_Causa.csv'.'
        employees: Merged DataFrame of employees_ and drops on 'id_contrato'.'
        df_: DataFrame containing raw employee data from 'Empleados_Activos_Retirados_V1.csv'.'
        df: Merged DataFrame of df_ and employees on 'id.1'.'
        identifiers: DataFrame containing identifier columns from df.'''
    df_inputs = {}
    df_inputs['employees_'] = pd.read_csv(os.path.join(input_path, 'Empleados_AR.csv'), encoding='latin-1', sep=';', low_memory=False)
    df_inputs['drops'] = pd.read_csv(os.path.join(input_path, 'Retiros_Causa.csv'), encoding='latin-1', low_memory=False)
    df_inputs['employees'] = df_inputs['employees_'].join(df_inputs['drops'].set_index('id_contrato'), on='id.1', how='inner', rsuffix='_')[['salario_mes', 'descripcion.4', 'id.1']]

    df_inputs['df_'] = pd.read_csv(
        os.path.join(input_path, 'Empleados_Activos_Retirados_V1.csv'),
        sep=';',
        encoding='latin-1',
        low_memory=False
    )
    df_inputs['df'] = df_inputs['df_'].join(df_inputs['employees'].set_index('id.1'), on='id.1', how='inner')

    df_inputs['identifiers'] = df_inputs['df'][cols['idents']]
    return df_inputs

def build_raw_data(df_inputs: dict) -> dict:
    ''' Build raw data by cleaning and merging input dataframes.
    Parameters
    ----------
    df_inputs : dict
        A dictionary containing the input dataframes.
    Returns
    -------
    dict
        A dictionary containing the cleaned and merged raw dataframes.
        df: Cleaned and merged DataFrame.'''
    df = df_inputs['df'].drop(cols['drop_cols'], axis=1)
    df = df.drop(cols['useless_cols'], axis=1)
    df = df.drop(cols['duplicated_cols'], axis=1)

    #Complementing databases with active employees
    complement = df_inputs['df_'][~df_inputs['df_']['id.1'].isin(df['id.1'])]                                                 #Filtering data to preserve active employees
    complement = complement[[col for col in df.columns if col in df_inputs['df_'].columns]]                                   #Selecting columns
    complement = complement.join(df_inputs['employees_'][['id.1', 'salario_mes']].set_index('id.1'), on='id.1', how='inner')  #Adding salary and contract columns                                                                  #Adding 'Activo' value to causa_retiro column
    df = pd.concat([df, complement])                                                                                          #Putting it together

    df = df.replace('1/01/2500', '1/01/1900')
    for column in cols['dates']:
        df[column] = df[column].str.replace(' 0:00', '')
        df[column] = pd.to_datetime(df[column], format='%d/%m/%Y')
    df = df.rename({
        'id.1': 'id_contrato',
        'fecha_ingreso.1': 'fecha_ingreso',
        'fecha_retiro.1': 'fecha_retiro',
        'descripcion.4': 'causa_retiro'
        }, axis=1
    )
    df['causa_retiro'] = df['causa_retiro'].fillna(value='Activo')
    df_inputs['df'] = df
    return df_inputs

def build_precurated_data(df_inputs: dict) -> dict:
    ''' Build precurated data by filtering and cleaning the raw data.
    Parameters
    ----------
    df_inputs : dict
        A dictionary containing the raw dataframes.
    Returns
    -------
    dict
        A dictionary containing the precurated dataframes.
        operative_stuff: Filtered and cleaned DataFrame for operative employees.
        days: Series containing the number of days each operative employee stayed in the company.'''
    operative_stuff = df_inputs['df'][
        (df_inputs['df'].Planta=='OPERATIVOS')&
        (df_inputs['df'].Proyecto.isin(cols['projects']))
    ]
    df_inputs['operative_stuff'] = operative_stuff.drop(cols['precurated_filter'], axis=1)
    days = (df_inputs['operative_stuff']['fecha_retiro']-df_inputs['operative_stuff']['fecha_ingreso']).dt.days
    df_inputs['days'] = days[(days>0)&(days<120)]
    return df_inputs

def define_base_line(df_inputs: dict) -> dict:
    ''' Define the baseline by creating a histogram of the days employees stayed in the
    company and calculating the standard deviation.
    Parameters
    ----------
    df_inputs : dict
        A dictionary containing the precurated dataframes.
    Returns
    -------
    plt.figure
        A matplotlib figure containing the histogram of days employees stayed in the company.'''
    desvest = round(float(df_inputs['days'].std()), 2)
    fig, ax = plt.subplots()
    counts, bins, _ = ax.hist(df_inputs['days'], bins=15, edgecolor='black', alpha=0.7)
    
    # Find the highest bar
    max_count = counts.max()
    bin_center = bins[-1]

    # Annotate the highest bar
    ax.annotate(
        f'Devest: {desvest} días',
        xy=(bin_center, max_count),
        xytext=(bin_center, max_count + 1),
        ha='right'
    )

    # Labels
    ax.set_title('Tiempos de permanencia')
    ax.set_xlabel('Días de permanencia en la empresa')
    ax.set_ylabel('Frecuencia')
    return fig

def save_preprocess(df_inputs: dict, fig: plt.figure, prefix: str='') -> None:
    ''' Save the preprocessed data and the baseline figure.
    Parameters
    ----------
    df_inputs : dict
        A dictionary containing the precurated dataframes.
    fig : plt.figure
        A matplotlib figure containing the histogram of days employees stayed in the company.
    prefix : str, optional
        Prefix to identify the files input-output. Like an unique identifier to make a trace between tests, by default '''''
    fig.savefig(os.path.join(output_path, 'base_line.png'), dpi=150, bbox_inches='tight')
    df_inputs['identifiers'].to_csv(
            os.path.join(output_path, 'databases', f'{prefix}_identifiers.csv'),
            index=0,
            sep=',',
            encoding='utf-8'
        )
    df_inputs['df'].to_csv(
        os.path.join(output_path, 'databases', f'{prefix}_raw_data.csv'),
        index=0,
        sep=',',
        encoding='utf-8'
    )
    df_inputs['operative_stuff'].to_csv(
        os.path.join(output_path, 'databases', f'{prefix}_precurated.csv'),
        index=0,
        sep=',',
        encoding='utf-8'
    )

def preprocess_data(prefix: str=''):
    ''' Preprocess the data by reading input files, building raw and precurated data,
    defining the baseline, and saving the results.
    Parameters
    ----------
    prefix : str, optional
        Prefix to identify the files input-output. Like an unique identifier to make a trace between tests, by default ''
    Returns
    -------
    tuple
        A tuple containing the precurated DataFrame and the baseline figure.'''
    print('process precurated data...')
    print('     reading data...')
    df_inputs = read_inputs()
    print('     building raw data...')
    df_inputs = build_raw_data(df_inputs)
    print('     building precurated data...')
    df_inputs = build_precurated_data(df_inputs)
    print('     defining baseline and saving results...')
    fig = define_base_line(df_inputs)
    save_preprocess(df_inputs, fig, prefix)
    return df_inputs, fig

if __name__=='__main__':
    df_inputs, fig = preprocess_data()