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

            start_row_1 = 326
            end_row_1 = 911
            start_row_2 = 1518
            end_row_2 = 2716






            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_tokenizer.add_special_tokens({"pad_token": "[PAD]"})


            # clear columns from NAN and float values
            df['Claim_text'].replace('', np.nan, inplace=True)
            df['Evidence_text'].replace('', np.nan, inplace=True)
            #df['Reason'].replace('', np.nan, inplace=True)
            df.dropna(subset=['Claim_text'], inplace=True)
            df.dropna(subset=['Evidence_text'], inplace=True)
            #df.dropna(subset=['Reason'], inplace=True)

            # Tokenize the 'claims' column
            '''
            df['tokenized_claims'] = df['Claim'].apply(
                lambda claim: bert_tokenizer.tokenize(claim)
            )
            '''

            df.loc[start_row_1:end_row_1, 'tokenized_claims'] = df.loc[start_row_1:end_row_1, 'Claim_text'].apply(
    lambda claim: bert_tokenizer.tokenize(claim)
)
            df.loc[start_row_2:end_row_2, 'tokenized_claims'] = df.loc[start_row_2:end_row_2, 'Claim_text'].apply(
    lambda claim: bert_tokenizer.tokenize(claim)
)


            # Filter out rows with empty tokenized claims
            #df = df[df['tokenized_claims'].apply(lambda x: len(x) > 0)]

            # Tokenize the 'evidence column'
            '''
            df['tokenized_evidences'] = df['Evidence'].apply(
                lambda evidence: bert_tokenizer.tokenize(evidence,)
            )
            '''

            df.loc[start_row_1:end_row_1, 'tokenized_evidences'] = df.loc[start_row_1:end_row_1, 'Evidence_text'].apply(
    lambda claim: bert_tokenizer.tokenize(claim)
)
            df.loc[start_row_2:end_row_2, 'tokenized_evidences'] = df.loc[start_row_2:end_row_2, 'Evidence_text'].apply(
    lambda claim: bert_tokenizer.tokenize(claim)
)
        


            #df = df[df['tokenized_evidences'].apply(lambda x: len(x) > 0)]

            # Tokenize the 'reason column'
            '''
            df['tokenized_reasons'] = df['Reason'].apply(
                lambda reason: bert_tokenizer.tokenize(reason)
            )
            '''

            '''
            df['tokenized_reason'] = df.loc[start_row:end_row, 'Reason'].apply(
    lambda claim: bert_tokenizer.tokenize(claim)
)
            '''


            '''
            df = df[df['tokenized_reasons'].apply(lambda x: len(x) > 0)]
            '''

            # Encode the tokenized content
            '''
            df['encoded_claims'] = df['Claim'].apply(
                lambda content : bert_tokenizer.encode_plus(content,
                return_tensors='tf', return_attention_mask=True,max_length=512, truncation=True,padding='max_length')
            )
            '''
            df.loc[start_row_1:end_row_1, 'encoded_claims'] = df.loc[start_row_1:end_row_1, 'Claim_text'].apply(
                lambda content : bert_tokenizer.encode_plus(content,
                return_tensors='tf', return_attention_mask=True,max_length=512, truncation=True,padding='max_length')
)
            df.loc[start_row_2:end_row_2, 'encoded_claims'] = df.loc[start_row_2:end_row_2, 'Claim_text'].apply(
                lambda content : bert_tokenizer.encode_plus(content,
                return_tensors='tf', return_attention_mask=True,max_length=512, truncation=True,padding='max_length')
)


            '''
            df['encoded_evidences'] = df['Evidence'].apply(
                lambda content : bert_tokenizer.encode_plus(content,
                return_tensors='tf', return_attention_mask=True,max_length=512, truncation=True,padding='max_length')
            )
            '''

            df.loc[start_row_1:end_row_1, 'encoded_evidences'] = df.loc[start_row_1:end_row_1, 'Evidence_text'].apply(
                lambda content : bert_tokenizer.encode_plus(content,
                return_tensors='tf', return_attention_mask=True,max_length=512, truncation=True,padding='max_length')
)
            df.loc[start_row_2:end_row_2, 'encoded_evidences'] = df.loc[start_row_2:end_row_2, 'Evidence_text'].apply(
                lambda content : bert_tokenizer.encode_plus(content,
                return_tensors='tf', return_attention_mask=True,max_length=512, truncation=True,padding='max_length')
)


            '''
            df['encoded_reasons'] = df['Reason'].apply(
                lambda content : bert_tokenizer.encode_plus(content,
                return_tensors='tf', return_attention_mask=True,max_length=512,truncation=True,padding='max_length')
            )
            '''

            '''
            df['encoded_reasons'] = df.loc[start_row:end_row, 'Reason'].apply(
                lambda content : bert_tokenizer.encode_plus(content,
                return_tensors='tf', return_attention_mask=True,max_length=512, truncation=True,padding='max_length')
)
            '''
            # Write the 2 columns (tokenized and encoded) to an Excel file
            df.to_excel(self.out_file)

            selected_rows = df[(df.index >= start_row_1) & (df.index <= end_row_1) | (df.index >= start_row_2) & (df.index <= end_row_2)]

            # Save the selected rows to a new Excel file
            selected_rows.to_excel('output_datasets/selected_rows.xlsx', index=False)

            return selected_rows
    



tm = TransformerModel('datasets/Masterfile.xlsx','output_datasets/modelset_final.xlsx')

pd_obj = tm.convert_pandas() 

transformed_data = tm.transform_data(pd_obj)


