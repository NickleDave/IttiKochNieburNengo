"""functions to convert feature maps into conspicuity channels and to combine those
to create the saliency "channel" S that feeds into the saliency map"""
import cv2
import numpy as np


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
    local_max = sum(local_max) / len(local_max)
    return local_max


def _normal(ftr_map, minmax=(0, 10)):
    """helper function that changes map range to minmax,
    by first normalize to range (0, 1), then multiplying by
    new range (i.e. new_range * some percent), and then adding
    the new minimum to all values"""
    # normalize map to range [0, 1]
    map_range = ftr_map.max() - ftr_map.min()
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
    I_bar = np.sum(norml_maps)
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

    C_bar = [np.sum([RG_norml_map, BY_norml_map])
             for RG_norml_map, BY_norml_map in zip(RG_norml_maps, BY_norml_maps)]

    C_bar = np.sum(C_bar)
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
    for map_start_ind in range(stop=len(O_maps), step=n_maps_per_theta):
        maps_this_theta = O_maps[map_start_ind : map_start_ind + n_maps_per_theta]
        norml_maps_this_theta = [normalize(map_, minmax, thresh, step)
                                 for map_ in maps_this_theta]
        norml_maps_this_theta = [cv2.resize(map_, size, interpolation=cv2.INTER_LINEAR)
                                 for map_ in norml_maps_this_theta]
        O_bar.append(np.sum(norml_maps_this_theta))

    O_bar = np.sum(O_bar)
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
    return (1 / 3) * np.sum(to_sum)
