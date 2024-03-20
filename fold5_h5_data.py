import os
import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import StratifiedKFold

current_directory = os.getcwd()
data_directory = os.path.join(current_directory, 'K_fold5_data')
os.makedirs(data_directory, exist_ok=True)
files = os.listdir(current_directory)
csv_files = [f for f in files if f.endswith('.csv')]

if len(csv_files) == 0:
    print("No CSV files found in the current directory.")
else:
    for csv_file in csv_files:
        csv_file_path = os.path.join(current_directory, csv_file)
        print(f"Processing {csv_file}...")

        train_df = pd.read_csv(csv_file_path)
        e = np.array(train_df['status'], dtype=np.int32)
        t = np.array(train_df['time'], dtype=np.float32)
        x = np.array(train_df.drop(['Tags', 'status', 'time'], axis=1), dtype=np.float32)
        e = np.nan_to_num(e, nan=0)
        t = np.nan_to_num(t, nan=0)
        x = np.nan_to_num(x, nan=0)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=47)
        fold = 1

        for train_index, test_index in skf.split(x, e):
            x_train, x_test = x[train_index], x[test_index]
            e_train, e_test = e[train_index], e[test_index]
            t_train, t_test = t[train_index], t[test_index]

            file_directory = os.path.join(data_directory, csv_file.replace('.csv', f'_fold_{fold}'))
            os.makedirs(file_directory, exist_ok=True)

            file_path = os.path.join(file_directory, f'{csv_file.replace(".csv", "_data.h5")}')

            h5file = h5py.File(file_path, 'w')

            group_train = h5file.create_group('train')
            group_test = h5file.create_group('test')

            group_train.create_dataset('e', data=e_train)
            group_train.create_dataset('t', data=t_train)
            group_train.create_dataset('x', data=x_train)

            group_test.create_dataset('e', data=e_test)
            group_test.create_dataset('t', data=t_test)
            group_test.create_dataset('x', data=x_test)

            h5file.close()

            fold += 1

print("Processing completed.")
