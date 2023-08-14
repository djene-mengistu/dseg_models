
import os
import logging
import sys
import warnings
import random
import numpy as np
warnings.filterwarnings("ignore")


#Creating the logging functions

def create_exp_dir(path, desc='Experiment dir: {}'):
    if not os.path.exists(path):
        os.makedirs(path)
    print(desc.format(path))


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_logger(log_dir):
    create_exp_dir(log_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'run.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger('Nas Seg')
    logger.addHandler(fh)
    return logger


#https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img):
    '''
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def make_mask(row_id, df):
    '''Given a row index, return image_id and mask (200, 200, 3) from the dataframe `df`'''
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:3]
    masks = np.zeros((200, 200, 3), dtype=np.float32) # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(200 * 200, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(200, 200, order='F')
    return fname, masks


