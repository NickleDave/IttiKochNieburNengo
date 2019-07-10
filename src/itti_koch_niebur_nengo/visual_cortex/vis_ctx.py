import cv2

from . import features
from . import saliency


def vis_ctx(img, sigma=8, c_range=(2, 3, 4), delta_range=(3, 4),
            thetas=(0, 45, 90, 135), ksize=(9, 9), gabor_sigma=4.0,
            lambd=10, gamma=0.5, psi=0, ktype=cv2.CV_32F,
            S_sigma=4, minmax=(0, 10), thresh=1, step=16
            ):
    """extract early features from image, create conspicuity channels
    from features maps, and then sum to create S, the input to saliency map"""
    # feature extraction
    R, G, B, Y, I = features.get_channels(img)
    I_maps = features.intensity_ftr_maps(I, sigma, c_range, delta_range)
    RG_maps, BY_maps = features.color_ftr_maps(R, G, B, Y,
                                               sigma, c_range, delta_range)
    O_maps = features.orientation_ftr_maps(I, thetas, sigma, c_range,
                                           delta_range, ksize, gabor_sigma,
                                           lambd, gamma, psi, ktype)

    # conspicuity channels, saliency computed from features
    height, width = I_maps[S_sigma].shape[:2]
    I_bar = saliency.I_conspicuity(I_maps, width, height, minmax, thresh, step)
    C_bar = saliency.C_conspicuity(RG_maps, BY_maps, width, height, minmax, thresh, step)
    O_bar = saliency.O_conspicuity(O_maps, width, height, minmax, thresh, step)
    S = compute_S(I_bar, C_bar, O_bar, minmax, thresh, step)

    return S
