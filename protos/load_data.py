import os
import pickle
from PIL import Image
import numpy
from multiprocessing import Pool

numpy.random.seed(0)

DATA_DIR = "../../data/data/"
TRAIN_DATA_DIR = DATA_DIR + "Train/"

WINDOW_SIZE = 50
MAP_IMAGE_SIZE = {(3328, 4992, 3): 458, (3744, 5616, 3): 488, (5616, 3744, 3): 2}

IMAGE_SIZE = (3744, 5616, 3)
USE_SIZE = (288, 432, 3)  # (936, 1404, 3)
SPLIT = int(IMAGE_SIZE[0] / USE_SIZE[0])


class SealionData:
    def __init__(self, idx, coord, s_type):
        self.idx = idx
        self.coord = coord
        self.s_type = s_type


try:
    with open('sealion_loc.pkl', 'rb') as f:
        list_lion = pickle.load(f)
except:
    list_lion = None


def split_batch(list_data, batch_size):
    ret = []
    for i in range(int(len(list_data) / batch_size) + 1):
        from_idx = i * batch_size
        next_idx = from_idx + batch_size if from_idx + batch_size <= len(list_data) else len(list_data)

        if from_idx >= next_idx:
            break

        ret.append(list_data[from_idx:next_idx])
    return ret


def load_data(batch):
    #p = Pool()
    tmp = list(map(_load_data, batch))
    # p.close()
    # p.join()
    data = [t[0] for t in tmp if t is not None]
    label = [t[1] for t in tmp if t is not None]
    return numpy.array(data), numpy.array(label)


def load_test_data(idx):
    img = numpy.array(Image.open(DATA_DIR + "Test/%s.jpg" % idx))
    if img.shape[0] > img.shape[1]:
        img = numpy.transpose(img, axes=(1, 0, 2))

    if img.shape != IMAGE_SIZE:
        if img.shape[0] > IMAGE_SIZE[0] or img.shape[1] > IMAGE_SIZE[1]:
            img = img[: IMAGE_SIZE[0], : IMAGE_SIZE[1], :]

        tmp = numpy.zeros(IMAGE_SIZE, dtype=numpy.int8)
        tmp[: img.shape[0], : img.shape[1], :] += img
        img = tmp
    ret = []
    for i in range(SPLIT):
        for j in range(SPLIT):
            ret.append(img[USE_SIZE[0] * i: USE_SIZE[0] * (i + 1),
                           USE_SIZE[1] * j: USE_SIZE[1] * (j + 1), :])

    return numpy.array(ret)


def concat_img(list_img):
    tmp = []
    for i in range(SPLIT):
        tmp.append(numpy.concatenate(list_img[i * SPLIT:(i + 1) * SPLIT], axis=1))
    return numpy.concatenate(tmp)


def _load_data2(idx):
    lion_data = list_lion[idx]
    if lion_data is None:
        return None
    img = numpy.array(Image.open(DATA_DIR + "Train/%s.jpg" % idx))
    if img.shape[0] > img.shape[1]:
        img = numpy.transpose(img, axes=(1, 0, 2))

    if img.shape != IMAGE_SIZE:
        tmp = numpy.zeros(IMAGE_SIZE, dtype=numpy.int8)
        tmp[: img.shape[0], : img.shape[1], :] += img
        img = tmp

    row_num = img.shape[0]
    col_num = img.shape[1]

    region = numpy.zeros((row_num, col_num, 1))

    for s_type, datas in lion_data.items():
        if s_type == 'error':
            continue
        for data in datas:
            y, x = data.coord
            region[x - WINDOW_SIZE: x + WINDOW_SIZE, y - WINDOW_SIZE: y + WINDOW_SIZE, :] = 1.

    return img, region


def _load_data(idx):
    lion_data = list_lion[idx]
    if lion_data is None:
        return None

    path = 'features/%s.pkl' % idx
    if os.path.exists(path):
        with open(path, 'rb') as f:
            img, region = pickle.load(f)
    else:
        img, region = _load_data2(idx)
        with open(path, 'wb') as f:
            pickle.dump((img, region), f, -1)

    x = numpy.random.randint(0, SPLIT)
    x_start = USE_SIZE[0] * x
    x_end = x_start + USE_SIZE[0]

    y = numpy.random.randint(0, SPLIT)
    y_start = USE_SIZE[1] * y
    y_end = y_start + USE_SIZE[1]

    img = img[x_start: x_end, y_start: y_end, :]
    region = region[x_start: x_end, y_start: y_end, :]

    return img, region
