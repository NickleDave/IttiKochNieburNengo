from pathlib import Path
import unittest

import cv2
import numpy as np

HERE = Path(__file__).parent
TEST_DATA = HERE.joinpath('test_data')
TEST_IMG_DIR = TEST_DATA.joinpath('img')


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.an_image_path = TEST_IMG_DIR.joinpath('test3.jpg')
        self.an_image = cv2.imread(self.an_image_path)

    def test_get_channels(self):
        R, G, B, Y, I = itti_koch_niebur_nengo.visual_cortex.features.get_channels(self.an_image)
        for channel in [R, G, B, Y, I]:
            self.assertTrue(
                type(channel) == np.ndarray
            )
            self.assertTrue(
                not np.any(channel < 0)
            )

    def test_gaussian_pyramid(self):
        R, G, B, Y, I = itti_koch_niebur_nengo.visual_cortex.features.get_channels(self.an_image)
        I_pyr = itti_koch_niebur_nengo.visual_cortex.features.gaussian_pyramid(I)
        self.assertTrue(len(I_pyr) == 8)  # default, from paper
        for level in I_pyr:
            self.assertTrue(type(level) is np.ndarray)

    def test_center_surround(self):
        R, G, B, Y, I = itti_koch_niebur_nengo.visual_cortex.features.get_channels(self.an_image)
        I_pyr = itti_koch_niebur_nengo.visual_cortex.features.gaussian_pyramid(I)
        I_maps = itti_koch_niebur_nengo.visual_cortex.features.center_surround(I_pyr)
        self.assertTrue(len(I_maps) == 6)  # default, from paper

if __name__ == '__main__':
    unittest.main()
