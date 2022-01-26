import numpy as np
import pandas as pd

def get_names(name, it):
    names = []
    for i in range(it):
        names.append(name+str(i))
    return names

def bin_addresses_deprocess(df, name):
    names = get_names(name, 32)
    intAdresses = pd.DataFrame()

    intAdresses[1] = df[names[0:8]].apply(
        lambda row: int(''.join(row.values.astype(str)), 2), axis=1)
    intAdresses[2] = df[names[8:16]].apply(
        lambda row: int(''.join(row.values.astype(str)), 2), axis=1)
    intAdresses[3] = df[names[16:24]].apply(
        lambda row: int(''.join(row.values.astype(str)), 2), axis=1)
    intAdresses[4] = df[names[24:32]].apply(
        lambda row: int(''.join(row.values.astype(str)), 2), axis=1)
    Adresses = pd.DataFrame()
    Adresses[name] = intAdresses.apply(
        lambda row: '.'.join(row.values.astype(str)), axis=1)

    print("deprocess " + name + " completed")

    return Adresses


def bin_port_deprocess(df, name):
    intPort = pd.DataFrame()
    intPort[name] = df[get_names(name, 16)].apply(
        lambda row: int(''.join(row.values.astype(str)), 2), axis=1)

    print("deprocess " + name + " completed")

    return intPort


def bin_pr_deprocess(df):
    names = get_names('pr', 8)

    intPr = pd.DataFrame()
    intPr['pr'] = df[names].apply(lambda row: int(
        ''.join(row.values.astype(str)), 2), axis=1)

    print("deprocess pr completed")

    return intPr['pr'].replace({1: 'ICMP', 6: 'TCP', 17: 'UDP', })


def bin_flg_deprocess(df):
    df_flg = (df['U'].replace({1: 'U', 0: '.', }) +
             df['A'].replace({1: 'A', 0: '.', }) +
             df['P'].replace({1: 'P', 0: '.', }) +
             df['R'].replace({1: 'R', 0: '.', }) +
             df['S'].replace({1: 'S', 0: '.', }) +
             df['F'].replace({1: 'F', 0: '.', }))
    df_flg.columns = ["flg"]
    return df_flg


def bin_ToS_deprocess(df):

    df = pd.DataFrame(df[get_names('stos', 8)].apply(lambda row: int(
        ''.join(row.values.astype(str)), 2), axis=1), columns=['stos'])

    print("deprocess stos completed")

    return df


def deprocess_discretized_distribution(df, name, thresholds):
    names = get_names(name, thresholds.size)
    deprocessed = pd.DataFrame()

    for i in range(thresholds.size):
        deprocessed[names[i]] = df[names[i]].replace(
            {1: thresholds[i], 0: 0, })

    print("deprocess " + name + " distribution completed")

    return pd.DataFrame(deprocessed.sum(axis=1), columns=[name])

def output_handler(df, columns):
    df = np.reshape( df, (df.shape[0]*75, df.shape[2]))
    df = pd.DataFrame(np.array(df))
    df.columns = columns
    df[df >= 0.5 ] = 1
    df[df < 0.5 ] = 0
    df = df.astype(int)
    return df


def df_deprocessing(df, thresholds):

    return pd.concat([ 
           deprocess_discretized_distribution(df[get_names('td',thresholds['td'].size)], 
                                              'td', thresholds['td']),
           bin_addresses_deprocess(df, 'sa'), 
           bin_addresses_deprocess(df, 'da'),
           bin_port_deprocess(df, 'sp'),
           bin_port_deprocess(df, 'dp'),
           bin_pr_deprocess(df),
           bin_flg_deprocess(df),
           df[['fwd']],
           bin_ToS_deprocess(df),
           deprocess_discretized_distribution(df[get_names('pkt',thresholds['pkt'].size)], 
                                              'pkt', thresholds['pkt']),
           deprocess_discretized_distribution(df[get_names('byt',thresholds['byt'].size)], 
                                              'byt', thresholds['byt']),     
            ], axis=1)    