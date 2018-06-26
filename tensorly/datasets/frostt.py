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
    """
    Get a dataset from FROSTT.

    Arguments
    ---------
    descriptor : str
        A descriptor that describes the FROSTT dataset. This is the filename
        without the extension ``tns.gz`` (e.g., to download the Amazon reviews
        dataset, specify ``descriptor = 'amazon-reviews'``).
    data_dir : str
        The directory to download the files into.

    Returns
    -------
    tensor : sparse.COO
        The sparse tensor. Note that this is a sparse.COO matrix and does not
        use the Tensorly backend.

    Notes
    -----
    Please cite the FROSTT paper if using any of these datasets, and look on
    frostt.io to see any other dataset-specific papers that should be cited.

    As of 2018-07, available ``descriptor``s are

    * ``amazon-reviews``
    * ``chicago-crime-comm``
    * ``chicago-crime-geo``
    * ``delicious-4d``
    * ``enron``
    * ``flickr-4d``
    * ``flickr-3d``
    * ``lbnl-network``
    * ``matmul_{m}-{n}-{p}`` where (m, n, p) in [(2,2,2), (3,3,3), (4,3,2), (4,4,3),
                                               (4,4,4), (5,5,5), (6,3,3)]
    * ``nell-1``
    * ``nell-2``
    * ``nips``
    * ``patents``
    * ``reddit-2015``
    * ``uber``
    * ``vast-2015-mc1-5d``
    * ``vast-2015-mc1-3d``

    References
    ----------
    @online{frosttdataset,
      title = {{FROSTT}: The Formidable Repository of Open Sparse Tensors and
               Tools},
      author = {Smith, Shaden and Choi, Jee W. and Li, Jiajia and Vuduc,
                Richard and Park, Jongsoo and Liu, Xing and Karypis, George},
      url = {http://frostt.io/},
      year = {2017},
    }
    """
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
