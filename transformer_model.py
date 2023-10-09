from transformers import BertTokenizer
from annotate import COVID_NAMES
import pandas as pd
import sys
import numpy as np


class TransformerModel:

    '''
    Recieve input file, and transform it to give output file
    '''
    
    
    def __init__(self, in_file:str, out_file:str):
        self.in_file = in_file
        self.out_file = out_file



    def convert_pandas(self):
        """
        Covert excel sheet to pandas object

        Args:
            file (str): excel file path
        """

        df = pd.read_excel(self.in_file)
        return df


    def transform_data(self,df):
            """
            tokenize and encode the data

            Returns:
                pandas.core.frame.DataFrame: Pandas dataframe with tokenized claims
            """


            
            # Initialize gpt2 tokenizer
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_tokenizer.add_special_tokens({"pad_token": "[PAD]"})


            # clear columns from NAN and float values
            df['Claim'].replace('', np.nan, inplace=True)
            df['Evidence'].replace('', np.nan, inplace=True)
            df['Reason'].replace('', np.nan, inplace=True)
            df.dropna(subset=['Claim'], inplace=True)
            df.dropna(subset=['Evidence'], inplace=True)
            df.dropna(subset=['Reason'], inplace=True)

            # Tokenize the 'claims' column
            df['tokenized_claims'] = df['Claim'].apply(
                lambda claim: bert_tokenizer.tokenize(claim)
            )

            # Filter out rows with empty tokenized claims
            df = df[df['tokenized_claims'].apply(lambda x: len(x) > 0)]

            # Tokenize the 'evidence column'
            df['tokenized_evidences'] = df['Evidence'].apply(
                lambda evidence: bert_tokenizer.tokenize(evidence,)
            )
        

            df = df[df['tokenized_evidences'].apply(lambda x: len(x) > 0)]

            # Tokenize the 'reason column'
            df['tokenized_reasons'] = df['Reason'].apply(
                lambda reason: bert_tokenizer.tokenize(reason)
            )

            df = df[df['tokenized_reasons'].apply(lambda x: len(x) > 0)]


            # Encode the tokenized content
            df['encoded_claims'] = df['Claim'].apply(
                lambda content : bert_tokenizer.encode_plus(content,
                return_tensors='tf', return_attention_mask=True,max_length=512, truncation=True,padding='max_length')
            )

            df['encoded_evidences'] = df['Evidence'].apply(
                lambda content : bert_tokenizer.encode_plus(content,
                return_tensors='tf', return_attention_mask=True,max_length=512, truncation=True,padding='max_length')
            )

            df['encoded_reasons'] = df['Reason'].apply(
                lambda content : bert_tokenizer.encode_plus(content,
                return_tensors='tf', return_attention_mask=True,max_length=512,truncation=True,padding='max_length')
            )

            # Write the 2 columns (tokenized and encoded) to an Excel file
            df.to_excel(self.out_file)

            return df


tm = TransformerModel('datasets/annotated_data.xlsx','output_datasets/modelset.xlsx')
pd_obj = tm.convert_pandas() 

transformed_data = tm.transform_data(pd_obj)