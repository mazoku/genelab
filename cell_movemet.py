import cv2
import matplotlib.pyplot as plt

import numpy as np

# Callback Function for Trackbar (but do not any work)
def nothing(*arg):
    pass

# Code here
def simple_trackbar(img, window_name='trackbar'):
    # Generate trackbar Window Name
    trackbar_name = window_name

    # Make Window and Trackbar
    cv2.namedWindow(window_name)
    cv2.createTrackbar(trackbar_name, window_name, 0, 255, nothing)

    # Allocate destination image
    threshold = np.zeros(img.shape, np.uint8)

    # Loop for get trackbar pos and process it
    while True:
        # Get position in trackbar
        trackbar_pos = cv2.getTrackbarPos(trackbar_name, window_name)
        # Apply threshold
        cv2.threshold(img, trackbar_pos, 255, cv2.THRESH_BINARY, threshold)
        # Show in window
        cv2.imshow(window_name, threshold)

        # If you press "ESC", it will return value
        ch = cv2.waitKey(5)
        if ch == 27:
            break

    cv2.destroyAllWindows()
    return threshold


def thresholding_test(img, t=100):
    # block_size = 11
    # c = 2
    # th1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
    # th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)

    # Otsu's thresholding
    # ret3, th3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    counts = []
    ts = range(5, 200, 5)
    cv2.namedWindow('output', flags=cv2.WINDOW_NORMAL)
    for t in ts:
        imt = (img > t).astype(np.uint8)
        n_pts = imt.sum()
        counts.append(n_pts)

        (cnts, _) = cv2.findContours(imt.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        clone = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cv2.drawContours(clone, cnts, -1, (0, 255, 0), 1)
        print 'T = {}, found {} contours'.format(t, len(cnts))

        cv2.imshow('output', clone)
        # cv2.waitKey(0)
        key = cv2.waitKey(0)
        key %= 256
        print key
        if key == 27:
            cv2.destroyAllWindows()
            break

    plt.figure()
    plt.plot(ts[:len(counts)], counts, '-o')
    plt.show()

    # plt.figure()
    # plt.subplot(121), plt.imshow(img, 'gray', interpolation='nearest')
    # plt.subplot(122), plt.imshow(th3, 'gray', interpolation='nearest')
    # # plt.subplot(133), plt.imshow(th3, 'gray', interpolation='nearest')
    # plt.show()


def analyze_particle(img, cnt, show=False, show_now=True):
    clone = img.copy()
    box = cv2.minAreaRect(cnt)
    box = np.int0(cv2.cv.BoxPoints(box))
    cv2.drawContours(clone, [cnt], -1, (0, 255, 0), 1)
    cv2.drawContours(clone, [box], -1, (0, 0, 255), 1)

    if show:
        # cv2.namedWindow('particle', flags=cv2.WINDOW_NORMAL)
        fig = plt.figure()
        fig.subplots_adjust(bottom=0.01, right=0.99, top=0.99, left=0.01)
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        ax2 = plt.subplot2grid((2, 3), (0, 2))
        ax3 = plt.subplot2grid((2, 3), (1, 2))

        for tl in ax1.get_xticklabels() + ax1.get_yticklabels():
            tl.set_visible(False)

        for tl in ax2.get_xticklabels() + ax2.get_yticklabels():
            tl.set_visible(False)

        for tl in ax3.get_xticklabels() + ax3.get_yticklabels():
            tl.set_visible(False)

        ax1.text(0.5, 0.5, 'ax1', va='center', ha='center')
        ax2.text(0.5, 0.5, 'ax2', va='center', ha='center')
        ax3.text(0.5, 0.5, 'ax3', va='center', ha='center')
        if show_now:
            plt.show()


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
def run(fname):
    img = cv2.imread(fname, 0)
    t = 10
    min_area = 4

    imt = img > t
    kernel_size = (3, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    imt = cv2.morphologyEx(imt.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    (cnts, _) = cv2.findContours(imt.astype(np.uint8).copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnts_f = [c for c in cnts if cv2.contourArea(c) >= min_area]

    analyze_particle(img, cnts_f[0], show=True)

    # clone1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # clone2 = clone1.copy()
    # cv2.drawContours(clone1, cnts, -1, (0, 255, 0), 1)
    # cv2.drawContours(clone2, cnts_f, -1, (0, 255, 0), 1)
    #
    # print 'original number = ', len(cnts)
    # print 'filtered number = ', len(cnts_f)
    #
    # cv2.namedWindow('original', flags=cv2.WINDOW_NORMAL)
    # cv2.imshow('original', clone1)
    # cv2.namedWindow('filtered', flags=cv2.WINDOW_NORMAL)
    # cv2.imshow('filtered', clone2)
    # while True:
    #     key = cv2.waitKey(10)
    #     key %= 256
    #     if key == 27:
    #         cv2.destroyAllWindows()
    #         break
    #
    # # simple_trackbar(img)
    # # thresholding_test(img)
    #
    # # cv2.namedWindow('input', flags=cv2.WINDOW_NORMAL)
    # # cv2.imshow('input', img)
    # # cv2.waitKey(0)


if __name__ == '__main__':
    fname = '/home/tomas/Dropbox/Data/genelab/Process_232_T0001.tif'

    run(fname)