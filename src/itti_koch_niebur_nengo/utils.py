"""utilify functions"""


def BGR2RGB(img):
    """swap channels of BGR image so it is RGB"""
    return img[:, :, [2, 1, 0]]