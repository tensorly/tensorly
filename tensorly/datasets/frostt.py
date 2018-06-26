import requests
from tqdm import tqdm
import os
import gzip

import numpy as np
import sparse

DATA_DIR = './_tensor-data/'


def download_file(url, local_path=DATA_DIR):
    local_filename = url.split('/')[-1]
    path = local_path + local_filename

    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))

    with open(path, 'wb') as f:
        chunk_size = 32 * 1024
        for chunk in tqdm(r.iter_content(chunk_size),
                          total=total_size / chunk_size, unit='B',
                          unit_scale=True):
            if chunk:
                f.write(chunk)

    return path


def frostt(descriptor, data_dir=DATA_DIR):
    if data_dir == DATA_DIR:
        try:
            os.makedirs(DATA_DIR)
        except FileExistsError:
            pass

    files = os.listdir(data_dir)
    if descriptor + '.tns.gz' in files:
        return read_dataset(data_dir + descriptor + '.tns.gz',
                            format=lambda coords, values: (coords - 1, values))

    prefix = 'https://s3.us-east-2.amazonaws.com/frostt/frostt_data/'
    url = prefix + descriptor + '/' + descriptor + '.tns.gz'
    download_file(url, local_path=data_dir)
    return frostt(descriptor, data_dir=data_dir)


def read_dataset(filename, format=None):
    with gzip.open(filename, 'rb') as f:
        raw = f.readlines()
    first_row = [float(x) for x in raw[0].split(b' ')]
    num_coords = len(first_row) - 1
    medium_rare = list(map(lambda line: line.strip(b'\n').split(b' '), raw))
    coords = (int(x) for line in medium_rare for x in line[:-1])
    values = (float(line[-1]) for line in medium_rare)

    coords = np.fromiter(coords, dtype=int).reshape(-1, num_coords)
    values = np.fromiter(values, dtype=float)
    if format:
        coords, values = format(coords, values)

    return sparse.COO(coords.T, data=values)
