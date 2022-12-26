import unittest
import pandas as pd
import numpy as np
from beton.model.exo_beton import dataSplitting, dataStandardisation


class TestDataPreprocessing(unittest.TestCase):
    def test_data_is_dataframe(self):
        self.assertIs(type(dataSplitting.X_train),pd.DataFrame)
    def test_data_is_numpy_ndarray(self):
        self.assertIs(type(dataStandardisation.scaled),np.ndarray)

if __name__ == '__main__':
    unittest.main()