"""functions to convert feature maps into conspicuity channels and to combine those
to create the saliency "channel" S that feeds into the saliency map"""
import warnings
warnings.filterwarnings("error")

import cv2
import numpy as np

from . import features


def find_avg_local_max(map, thresh=1, step=16):
    """find average local maximum across feature map"""
    local_max = []
    height, width = map.shape[:2]
    for y in range(0, height - step, step):
        for x in range(0, width - step, step):
            local_img = map[y:y + step, x:x + step]
            lmin, lmax, _, _ = cv2.minMaxLoc(local_img)
            if lmax > thresh:
                local_max.append(lmax)

    # to avoid "divide by zero" error
    # when Nengo tests function while adding to network
    if len(local_max) > 0:
        local_max = sum(local_max) / len(local_max)
    else:
        return 0
    return local_max


def _normal(ftr_map, minmax=(0, 10)):
    """helper function that changes map range to minmax,
    by first normalize to range (0, 1), then multiplying by
    new range (i.e. new_range * some percent), and then adding
    the new minimum to all values"""
    # normalize map to range [0, 1]
    map_range = ftr_map.max() - ftr_map.min()

    if map_range == 0:
        pass  # don't do anything, because there's no range to normalize
    else:
        ftr_map = (ftr_map - ftr_map.min()) / map_range
        # then convert to desired range
        new_range = minmax[1] - minmax[0]
        ftr_map = (ftr_map * new_range) + minmax[0]

    return ftr_map


def normalize(ftr_map, minmax=(0, 10), thresh=1, step=16):
    """Normalize feature map by multiplying it with
    (max(map) - avg(local_maxima))^2
    as described in Itti Koch Neibur 1998

    Parameters
    ----------
    ftr_map : numpy.ndarray
    minmax : tuple
    thresh : int
    step : int

    Returns
    -------

    Notes
    -----
    Based in part on the Saliency Toolbox implementation,
    as inferred from Jonathan Harel's code
    http://www.vision.caltech.edu/~harel/share/gbvs.php
    """
    ftr_map = _normal(ftr_map, minmax)
    avg_local_max = find_avg_local_max(ftr_map, thresh, step)
    return ftr_map * (ftr_map.max() - avg_local_max) ** 2


def I_conspicuity(I_maps, width, height, minmax=(0, 10), thresh=1, step=16):
    """

    Parameters
    ----------
    I_maps
    width
    height
    minmax
    thresh
    step

    Returns
    -------
    I_bar
    """
    size = (width, height)

    norml_maps = [normalize(map_, minmax, thresh, step)
                  for map_ in I_maps]
    norml_maps = [cv2.resize(map_, size, interpolation=cv2.INTER_LINEAR)
                  for map_ in norml_maps]
    I_bar = np.sum(norml_maps, axis=0)
    return I_bar


def C_conspicuity(RG_maps, BY_maps, width, height, minmax=(0, 10), thresh=1, step=16):
    """

    Parameters
    ----------
    RG_maps
    BY_maps
    width
    height
    minmax
    thresh
    step

    Returns
    -------
    C_bar
    """
    size = (width, height)

    RG_norml_maps = [normalize(map_, minmax, thresh, step)
                     for map_ in RG_maps]
    RG_norml_maps = [cv2.resize(map_, size, interpolation=cv2.INTER_LINEAR)
                     for map_ in RG_norml_maps]

    BY_norml_maps = [normalize(map_, minmax, thresh, step)
                     for map_ in BY_maps]
    BY_norml_maps = [cv2.resize(map_, size, interpolation=cv2.INTER_LINEAR)
                     for map_ in BY_norml_maps]

    C_bar = [np.sum([RG_norml_map, BY_norml_map], axis=0)
             for RG_norml_map, BY_norml_map in zip(RG_norml_maps, BY_norml_maps)]

    C_bar = np.sum(C_bar, axis=0)
    return C_bar


def O_conspicuity(O_maps, width, height, n_maps_per_theta=6,
                  minmax=(0, 10), thresh=1, step=16):
    """

    Parameters
    ----------
    O_maps
    width
    height
    n_maps_per_theta
    minmax
    thresh
    step

    Returns
    -------
    O_bar
    """
    size = (width, height)

    O_bar = []
    for map_start_ind in range(0, len(O_maps), n_maps_per_theta):
        maps_this_theta = O_maps[map_start_ind : map_start_ind + n_maps_per_theta]
        norml_maps_this_theta = [normalize(map_, minmax, thresh, step)
                                 for map_ in maps_this_theta]
        norml_maps_this_theta = [cv2.resize(map_, size, interpolation=cv2.INTER_LINEAR)
                                 for map_ in norml_maps_this_theta]
        O_bar.append(np.sum(norml_maps_this_theta, axis=0))

    O_bar = np.sum(O_bar, axis=0)
    return O_bar


def compute_S(I_bar, C_bar, O_bar, minmax=(0, 10), thresh=1, step=16):
    """compute S, the input to the saliency map.
    Each conspicuity ch

    Parameters
    ----------
    I_bar : numpy.ndarray
        returned by saliency.I_conspicuity
    C_bar : numpy.ndarray
        returned by saliency.C_conspicuity
    O_bar : numpy.ndarray
        returned by saliency.O_conspicuity

    Returns
    -------
    S : numpy.ndarray
    """
    to_sum = []
    for conspicuity_channel in [I_bar, C_bar, O_bar]:
        to_sum.append(
            normalize(conspicuity_channel, minmax, thresh, step)
        )
    return (1 / 3) * np.sum(to_sum, axis=0)


def img_to_S(img, sigma=8, c_range=(2, 3, 4), delta_range=(3, 4),
             thetas=(0, 45, 90, 135), ksize=(9, 9), gabor_sigma=4.0,
             lambd=10, gamma=0.5, psi=0, ktype=cv2.CV_32F,
             S_sigma=3, minmax=(0, 10), thresh=1, step=16
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
    I_bar = I_conspicuity(I_maps, width, height, minmax, thresh, step)
    C_bar = C_conspicuity(RG_maps, BY_maps, width, height, minmax, thresh, step)
    n_maps_per_theta = len(c_range) * len(delta_range)
    O_bar = O_conspicuity(O_maps, width, height, n_maps_per_theta,
                          minmax, thresh, step)
    S = compute_S(I_bar, C_bar, O_bar, minmax, thresh, step)

    return S
