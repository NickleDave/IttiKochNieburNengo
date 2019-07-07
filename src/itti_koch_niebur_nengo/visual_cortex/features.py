"""extract early visual features for Itti Koch Niebur model"""
import cv2
import numpy as np


def gaussian_pyramid(img, sigma=8):
    """compute gaussian pyramid representation of image
    
    Parameters
    ----------
    img : numpy.ndarray
        image, loaded using cv2.
    sigma : int
        number of levels in gaussian pyramid.
        Default is 8, the number used in Itti Koch Nieber.
        Level zero is the original image,
        levels 1 through sigma are the subsampled and smoothed of the level below them.
    
    Returns
    -------
    img_out : list
        of numpy.ndarrays, with sigma + 1 elements.
    """
    img_out = [img]
    for i in range(sigma):
        img_out.append(cv2.pyrDown(img_out[i]))
    return img_out


def center_surround(pyramid_c, pyramid_s=None,
                    c_range=(2, 3, 4), delta_range=(3, 4)):
    """compute center-surround difference from a gaussian pyramid
    and return feature maps
    
    Parameters
    ----------
    pyramid_c : list
        of numpy.ndarray, returned by gaussian_pyramid.
        Centers of receptive fields are taken from levels of this pyramid.
    pyramid_s : list
        of numpy.ndarray, returned by gaussian_pyramid.
        Surrounds of receptive fields are taken from levels of this pyramid.
        Default is None, in which case pyramid_c is used.
    c_range : tuple, list
        of int, indices of levels of pyramid to use as "centers".
        Default is (2, 3, 4), levels used in Itti Koch Niebur.
    delta_range : tuple, list
        of int, indices of levels of pyramid to use as "surrounds".
        Default is (3, 4), levels used in Itti Koch Niebur.

    Returns
    -------
    maps : list
        of numpy.ndarray
    """
    if pyramid_s is None:
        pyramid_s = pyramid_c

    maps = []
    for delta in delta_range:
        for c in c_range:
            center = pyramid_c[c]
            surround = pyramid_s[c + delta]
            surround = cv2.resize(surround, center.shape[:2],
                                  interpolation=cv2.INTER_LINEAR)
            maps.append(cv2.absdiff(center, surround))
    return maps


def get_channels(img):
    """take RGB image and convert to 
    red, green, blue, yellow and intensity channels
    used by Itti Koch Niebur model
    
    Parameters
    ----------
    img : numpy.ndarray
        returned by cv2.imread
    
    Returns
    -------
    R, G, B, Y, I : numpy.ndarray
        calculated as described in Itti Koch Niebur
    """
    b, g, r = cv2.split(img)
    I = (b + g + r) / 3

    # normalize r g b by I
    # to "decouple hue from intensity"
    # only do so at locations where intensity is greater than 1/10 of its maximum
    I_norml = np.where(I > 0.1 * I.max(), I, 1)
    b = b / I_norml
    g = g / I_norml
    r = r / I_norml
    
    R = r - (g + b) / 2
    G = g - (r + b) / 2
    B = b - (r + g) / 2
    Y = (r + g) / 2 - np.abs(r - g) / 2 - b
    colors = [R, G, B, Y]
    # negative values are set to zero
    (R, G, B, Y) = map(lambda arr: np.where(arr >= 0, arr, 0), 
                       colors)

    return R, G, B, Y, I


def intensity_ftr_maps(I, sigma=8, c_range=(2, 3, 4), delta_range=(3, 4)):
    """compute intensity feature maps, given intensity channel
    
    Parameters
    ----------
    I : numpy.ndarray
        intensity channel, returned by get_channels
    sigma : int
        parameter for gaussian_pyramid function, number of levels.
        Default is 8.
    c_range : list, tuple
        of int. Parameter for center_surround function, levels to use
        as centers. 
        Default is (2, 3, 4)
    delta_range : list, tuple
        of int. Parameter for center_surround function, used to find
        surround levels s. For each center c there will be a surround s 
        which is (c + delta).
        Default is (3, 4)

    Returns
    -------
    maps : list
        of numpy.ndarray, maps computed by extracting gaussian pyramid
    """
    maps = []
    I_pyr = gaussian_pyramid(I, sigma=sigma)
    maps.extend(
        center_surround(I_pyr, c_range=c_range, delta_range=delta_range)
    )
    
    I_inverse = cv2.bitwise_not(I)
    I_inv_pyr = gaussian_pyramid(I_inverse, sigma=sigma)
    maps.extend(
        center_surround(I_inv_pyr, c_range=c_range, delta_range=delta_range)
    )
    return maps


def color_ftr_maps(R, G, B, Y,
                   sigma=8, c_range=(2, 3, 4), delta_range=(3, 4)):
    """compute color feature maps, given color channels
    
    Parameters
    ----------
    R, G, B, Y : numpy.ndarray
        color channels, returned by get_channels
    sigma : int
        parameter for gaussian_pyramid function, number of levels.
        Default is 8.
    c_range : list, tuple
        of int. Parameter for center_surround function, levels to use
        as centers. 
        Default is (2, 3, 4)
    delta_range : list, tuple
        of int. Parameter for center_surround function, used to find
        surround levels s. For each center c there will be a surround s 
        which is (c + delta).
        Default is (3, 4)

    Returns
    -------
    maps : list
        of numpy.ndarray, maps computed by extracting gaussian pyramid
    """
    maps = []

    R_pyr, G_pyr, B_pyr, Y_pyr = map(
        lambda c: gaussian_pyramid(c, sigma=sigma),
        [R, G, B, Y]
    )

    # paper represents color opponency in receptive fields as follows:
    # RG = (R(c) - G(c)) center-surround diff (G(s) - R(s))
    RG_c = [
        r_c - g_c
        for r_c, g_c in zip(R_pyr, G_pyr)
    ]
    GR_s = [
        g_s - r_s
        for r_s, g_s in zip(R_pyr, G_pyr)
    ]
    maps.extend(
        center_surround(pyramid_c=RG_c, pyramid_s=GR_s,
                        c_range=c_range, delta_range=delta_range)

    )

    # do same thing for blue-yellow
    BY_c = [
        b_c - y_c
        for b_c, y_c in zip(B_pyr, Y_pyr)
    ]
    YB_s = [
        y_s - b_s
        for b_s, y_s in zip(B_pyr, Y_pyr)
    ]
    maps.extend(
        center_surround(pyramid_c=BY_c, pyramid_s=YB_s,
                        c_range=c_range, delta_range=delta_range)

    )
    return maps
