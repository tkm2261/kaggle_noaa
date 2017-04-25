import os
import numpy
from load_data import split_batch, load_test_data, concat_img

import glob
from train_unet import get_unet
DATA_DIR = "../../data/data/"
TRAIN_DATA_DIR = DATA_DIR + "Train/"
TEST_DATA_DIR = DATA_DIR + "Test/"

SMOOTH = 1e-12
CLASS_NUM = 1

BATCH_SIZE = 50

MODEL = 'weights/unet_tmp.hdf5'


def test():
    model = get_unet()
    model.load_weights(MODEL)

    for path in glob.glob(TEST_DATA_DIR + '*.jpg'):
        print(path)
        idx = int(os.path.basename(path).split('.')[0])
        img = load_test_data(idx)
        return model.predict(img, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    aaa = test()
    preds = concat_img(aaa)
    numpy.savetxt('tmp.npy', preds)
