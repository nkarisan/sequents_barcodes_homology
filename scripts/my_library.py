"""
Author: Negin Karisani
Date: May 2022
"""
from datetime import datetime
import pandas as pd
import numpy as np
import os
import sys
import pickle
from scipy import stats
import h5py
import socket


class MyLib:
    save_dir = '/Users/nkarisan/PycharmProjects/sequents_barcodes_homology/output_files/'

    try:
        h_name = socket.gethostname()
        if h_name.startswith('gilbreth'):
            root = "/scratch/gilbreth/nkarisan/BI/cach/data"
        elif h_name.startswith('bell'):
            root = '/scratch/bell/nkarisan/BI/depmap_files/'
        else:
            root = "input_files/"
    except Exception as e:
        print('Exception ', e)
        sys.exit(1)

    @staticmethod
    def get_time():
        return datetime.now().strftime('%Y/%m/%d %H:%M:%S')

    @staticmethod
    def load_csv(file_path, columns=None, index_col=None, **kwargs):
        """
            sample argument:  **{'header': None})
        """
        try:
            data_df = pd.read_csv(os.path.join(MyLib.root, file_path), usecols=columns, index_col=index_col, **kwargs)
            print(os.path.join(MyLib.root, file_path), " is loaded, shape: ", data_df.shape)
            print()
            return data_df
        except OSError:
            print("Could not open/read file:", os.path.join(MyLib.root, file_path))
            sys.exit()

    @staticmethod
    def save_hdf5(data_df, file_path):
        dest = h5py.File(os.path.join(MyLib.save_dir, file_path), 'w')

        try:
            dim_0 = [x.encode('utf8') for x in data_df.index]
            dim_1 = [x.encode('utf8') for x in data_df.columns]

            dest.create_dataset('dim_0', track_times=False, data=dim_0)
            dest.create_dataset('dim_1', track_times=False, data=dim_1)
            dest.create_dataset("data", track_times=False, data=data_df.values)
            print('\nFile ' + os.path.join(MyLib.save_dir, file_path), 'saved, data shape: ', data_df.shape)
            print()
        finally:
            dest.close()

    @staticmethod
    def load_h5py(file_path):
        src = h5py.File(os.path.join(MyLib.root, file_path), 'r')
        try:
            dim_0 = [x.decode('utf8') for x in src['dim_0']]
            dim_1 = [x.decode('utf8') for x in src['dim_1']]
            data = np.array(src['data'])
            print(os.path.join(MyLib.root, file_path), " is loaded, shape: ", data.shape)
            print()
            return pd.DataFrame(index=dim_0, columns=dim_1, data=data)
        finally:
            src.close()

    @staticmethod
    def save_pkl(data, file_path):
        with open(os.path.join(MyLib.save_dir, file_path), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            print('\nFile ' + os.path.join(MyLib.save_dir, file_path), 'saved, data size: ', len(data))
            print()

    @staticmethod
    def load_pkl(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print('\nFile ', file_path, " is loaded, data type:", type(data),
                  " size: ", len(data))
            print()
            return data

    @staticmethod
    def compute_correlations(df1, df2):
        """
            computes Pearson's correlation coefficient between columns of df1 and df2
        """
        stat_val_df = pd.DataFrame(index=df1.columns)
        for col in df2.columns:
            stat_val_df[col] = df1.corrwith(df2[col], method=lambda x, y: stats.pearsonr(x, y)[0]).values

        return stat_val_df