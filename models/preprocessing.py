import numpy as np
import pandas as pd
import os
import glob
import time
import tarfile
import ipaddress


def read_multiple_csv(path):

    dt = []

    all_files = glob.iglob(os.path.join(path, "*.csv"))

    for f in all_files:
        try:
            df = pd.read_csv(f, header=None, error_bad_lines=False)
            dt.append(df)
        except:
            continue
    df = pd.concat(dt, axis=0)
    
    df.columns = ['te', 'td', 'sa', 'da', 'sp', 'dp',
                  'pr', 'flg', 'fwd', 'stos', 'pkt', 'byt', 'label']

    return df


def get_names(name, it):
    names = []
    for i in range(it):
        names.append(name+str(i))
    return names


def bin_port(df, name):
    df = pd.DataFrame(df.apply(lambda port: bin(int(port))[2:].zfill(16)))
    df = pd.DataFrame(df.to_numpy().astype('U16').view('U1').astype(int))

    names = []
    for i in range(16):
        names.append(name+str(i))

    df.columns = names
    print("bin_port " + name + " completed")
    return df


def bin_addresses(df, name):
    df = pd.DataFrame(df.apply(lambda ip: bin(
        int(ipaddress.ip_address(ip)))[2:].zfill(32)))
    df = pd.DataFrame(df.to_numpy().astype('U32').view('U1').astype(int))

    names = []
    for i in range(32):
        names.append(name+str(i))

    df.columns = names
    print("bin_adress " + name + " completed")
    return df


def get_bin_pr(hex_value):
    return bin(hex_value)[2:].zfill(8)


def bin_pr(df):     # Este método convierte el protocolo a su representación binaria
    #  actualmente solo se necesita ICM, TCP y UDP
    df = df.replace({'ICMP': get_bin_pr(0X01), 'TCP': get_bin_pr(
        0X06), 'UDP': get_bin_pr(0X11), })
    df = pd.DataFrame(df.str.split('', n=8, expand=True).drop(columns=[0]))

    names = []
    for i in range(8):
        names.append('pr'+str(i))

    df.columns = names
    print("bin_pr completed")
    return df


def bin_flag(df):  # UAPRSF dividir en 6 columnas, si es un punto 0 si es otra cosa 1
    df = pd.DataFrame(df.str.split('', n=6, expand=True)).drop(columns=[0]
                                                               ).replace({'.': 0, 'U': 1, 'A': 1, 'P': 1, 'R': 1, 'S': 1, 'F': 1, })
    df.columns = ['U', 'A', 'P', 'R', 'S', 'F']
    print("bin_flag completed")
    return df


def bin_ToS(df, name):
    df = pd.DataFrame(df.apply(lambda ToS: bin(int(ToS))[2:].zfill(8)))
    df = pd.DataFrame(df.to_numpy().astype('U8').view('U1').astype(int))

    names = []

    for i in range(8):
        names.append(name+str(i))

    df.columns = names
    print("bin_ToS completed")
    return df


def get_bin_value(size, position):
    value = np.zeros(size, dtype=int)
    value[position] = 1
    return value


def get_distribution(df):
    data = df.describe()
    i = data['min']  
    maximun = data['max']
    
    if i != data['25%']:
        result = np.array([i,data['25%']])
    else: result = np.array([data['25%']])
        
    while i/maximun <= 0.9 and data['75%'] != data['50%']:
        if result[-1] != data['75%']:
            result = np.append(result,[data['75%']])
        i = data['75%']
        data = df[(df >= i)].describe()
        
    if result[-1] != maximun:
        result = np.append(result,maximun)
    
    return result

def get_pkt_distribution(df):
    data = df.describe()
    maximun = data['max']
    
    result = np.array([])
    
    for i in range(0,30):
        result = np.append(result,i)
        
    for i in range(30,int(maximun), 50):
        result = np.append(result,i)
    
    if result[-1] != maximun:
        result = np.append(result,maximun)
    
    return result

def get_byt_distribution(df):
    data = df.describe()
    
    result = np.array([])
    
    for i in range(int(data['min']),300, 7):
        result = np.append(result,i)
    
    for i in range(300,600, 50):
        result = np.append(result,i)

    for i in range(600,2000, 300):
        result = np.append(result,i)
    
    if result[-1] != data['max']:
        result = np.append(result,data['max'])
    
    return result

def discretize_distribution(df, name, thresholds):
    
    names = get_names(name,thresholds.size)
    
    result = np.zeros((df.size,thresholds.size), dtype=int)
    df = np.array(df)
    
    for i in range(thresholds.size-1):
        result[np.where((df > thresholds[i]) & (df <= thresholds[i+1])) ] = get_bin_value(thresholds.size,i)
    
    return pd.DataFrame(result,columns = names),{ name: thresholds}   


def df_preprocessing(df, thresholds):
    # Esta función utiliza todas las funciones anteriormente declaradas
    # para preprocesar los datos a utilizar con los modelos de IA
    
    df = df.reset_index()
    
    td_dist, td_thres = discretize_distribution(df['td'], 'td', thresholds['td'])
    print("td discretized")
    pkt_dist, pkt_thres = discretize_distribution(df['pkt'], 'pkt', thresholds['pkt'])
    print("pkt discretized")
    byt_dist, byt_thres = discretize_distribution(df['byt'], 'byt', thresholds['byt'])
    print("byt discretized")

    return pd.concat([
                     df['te'],
                     td_dist,
                     bin_addresses(df['sa'], 'sa'),
                     bin_addresses(df['da'], 'da'),
                     bin_port(df['sp'], 'sp'),
                     bin_port(df['dp'], 'dp'),
                     bin_pr(df['pr']),
                     bin_flag(df['flg']),
                     df['fwd'],
                     bin_ToS(df['stos'], 'stos'),
                     pkt_dist,
                     byt_dist,
                     df['label']],
                     axis='columns'), thresholds


def create_dataframes(df):
    df = df.drop(['te'], axis=1)
    df = df.drop(['label'], axis=1)
    df = np.int8(df)
    df = df[:(df.shape[0] - df.shape[0]%75)]
    df = np.reshape(df,(df.shape[0]//75,75,df.shape[1]))
    return df


def shuffle(df, test_proportion):
    ratio = int(df.shape[0]/test_proportion)
    X_train = df[ratio:]
    X_test =  df[:ratio]
    return X_train, X_test