import os
import matplotlib.pyplot as plt
import pickle
import pandas
import numpy
from PIL import Image
import skimage.feature
import pickle
import cv2
from logging import getLogger
from multiprocessing import Pool
logger = getLogger(__name__)

from load_data import SealionData, DATA_DIR, TRAIN_DATA_DIR
MAX_AVG_DIFF = 50

SET_MISMATCH_IDS = set(pandas.read_csv(DATA_DIR + 'MismatchedTrainImages.txt')['train_id'].values.tolist())


def find_data(idx):

    if idx in SET_MISMATCH_IDS:
        logger.info('Mismatched %s' % idx)
        return None

    logger.info("{}".format(idx))
    map_count = {'adult_males': [],
                 'subadult_males': [],
                 'adult_females': [],
                 'juveniles': [],
                 'error': [],
                 'pups': []}

    # read the Train and Train Dotted images
    image_1 = cv2.imread(DATA_DIR + "TrainDotted/{}.jpg".format(idx))
    image_2 = cv2.imread(DATA_DIR + "Train/{}.jpg".format(idx))
    # absolute difference between Train and Train Dotted
    image_3 = cv2.absdiff(image_1, image_2)

    if idx in SET_MISMATCH_IDS:
        logger.info('Mismatched')
        return None

    # mask out blackened regions from Train Dotted
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 20] = 0
    mask_1[mask_1 > 0] = 255

    mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    mask_2[mask_2 < 20] = 0
    mask_2[mask_2 > 0] = 255

    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_2)

    # convert to grayscale to be accepted by skimage.feature.blob_log
    image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)

    # detect blobs
    blobs = skimage.feature.blob_log(image_3, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)
    for blob in blobs:
        # get the coordinates for each blob
        y, x, s = blob
        # get the color of the pixel from Train Dotted in the center of the blob
        b, g, r = image_1[int(y)][int(x)][:]

        # decision tree to pick the class of the blob by looking at the color in Train Dotted
        if r > 200 and b < 50 and g < 50:  # RED
            s_type = "adult_males"
        elif r > 200 and b > 200 and g < 50:  # MAGENTA
            s_type = "subadult_males"
        elif r < 100 and b < 100 and 150 < g < 200:  # GREEN
            s_type = "pups"
        elif r < 100 and 100 < b and g < 100:  # BLUE
            s_type = "juveniles"
        elif r < 150 and b < 50 and g < 100:  # BROWN
            s_type = "adult_females"
        else:
            s_type = "error"

        map_count[s_type].append(SealionData(idx, (int(x), int(y)), s_type))

    return map_count


if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler(os.path.abspath(__file__) + '.log', 'w')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    df = pandas.read_csv(TRAIN_DATA_DIR + 'train.csv')
    p = Pool()
    list_result = list(p.map(find_data, df['train_id'].values))
    p.close()
    p.join()
    with open('sealion_loc.pkl', 'wb') as f:
        pickle.dump(list_result, f, -1)
