import unittest
import pandas as pd

from transformer_model import TransformerModel

class TestTransformerModel(unittest.TestCase):
    def setUp(self):
        self.in_file = 'datasets/Masterfile.xlsx'
        self.out_file = 'output_datasets/modelset_final.xlsx'
        self.transformer_model = TransformerModel(self.in_file, self.out_file)

    def test_convert_pandas(self):
        df = self.transformer_model.convert_pandas()
        self.assertIsInstance(df, pd.DataFrame)

    def test_transform_data(self):
        pd_obj = self.transformer_model.convert_pandas()
        selected_rows = self.transformer_model.transform_data(pd_obj)
        self.assertIsInstance(selected_rows, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()