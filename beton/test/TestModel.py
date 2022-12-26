import unittest
from tensorflow import keras
from beton.model.exo_beton import dataModeling 

class TestModel(unittest.TestCase):
    def test_if_model_is_keras_engine_sequential(self):
        self.assertIs(dataModeling.model.__module__,keras.Sequential.__module__)

if __name__ == '__main__':
    unittest.main()

