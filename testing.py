import unittest
import pandas as pd

from transformer_model import TransformerModel

class TestTransformerModel(unittest.TestCase):
    def setUp(self):
        # Set up any common objects or configurations needed for the tests
        self.in_file = 'datasets/Masterfile.xlsx'
        self.out_file = 'output_datasets/modelset_final.xlsx'
        self.transformer_model = TransformerModel(self.in_file, self.out_file)

    def test_convert_pandas(self):
        # Test the 'convert_pandas' method
        df = self.transformer_model.convert_pandas()
        self.assertIsInstance(df, pd.DataFrame)
        # Add more assertions as needed

    def test_transform_data(self):
        # Test the 'transform_data' method
        pd_obj = self.transformer_model.convert_pandas()
        selected_rows = self.transformer_model.transform_data(pd_obj)
        self.assertIsInstance(selected_rows, pd.DataFrame)
        # Add more assertions as needed

    # Add more test cases for other methods as needed

if __name__ == '__main__':
    unittest.main()