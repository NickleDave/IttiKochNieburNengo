import cv2
import numpy as np

from . import saliency


def make_vis_ctx(img_shape, sigma=8, c_range=(2, 3, 4), delta_range=(3, 4),
                 thetas=(0, 45, 90, 135), ksize=(9, 9), gabor_sigma=4.0,
                 lambd=10, gamma=0.5, psi=0, ktype=cv2.CV_32F,
                 S_sigma=3, minmax=(0, 10), thresh=1, step=16):
    """extract early features from image, create conspicuity channels
    from features maps, and then sum to create S, the input to saliency map"""
    # feature extraction
    def img_to_S_with_params_fixed():
        def anon_img_to_S(img):
            return saliency.img_to_S(img, sigma, c_range, delta_range,
                                     thetas, ksize, gabor_sigma,
                                     lambd, gamma, psi, ktype,
                                     S_sigma, minmax, thresh, step)
        return anon_img_to_S

    img_to_S_params_fixd = img_to_S_with_params_fixed()

    def vis_ctx():
        def anon_vis_ctx(x):
            img = np.reshape(x, img_shape)
            S = img_to_S_params_fixd(img)
            return S.flatten()
        return anon_vis_ctx

    return vis_ctx()

