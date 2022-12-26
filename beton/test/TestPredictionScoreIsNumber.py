import unittest
import pickle
import numpy as np


compute_r2_score = pickle.load(open('compute_r2_score.pkl', 'rb'))


class TestPrediction(unittest.TestCase):
    def test_prediction_score_is_float64(self):
        self.assertIs(type(compute_r2_score),np.float64)


if __name__ == '__main__':
    unittest.main()