from transformers import GPT2Tokenizer
from annotate import filter_by_covid, COVID_NAMES
import torch
import pandas as pd




def transform_data(df, output_file_name="output_datasets/finalver1_complete.xlsx"):
    """
    Tokenize, Encode,  and Decode data.

    Args:
        df (pandas.core.frame.DataFrame): Pandas dataframe to be tokenized
        output_file_name (str): The file name to write the tokenized claims

    Returns:
        pandas.core.frame.DataFrame: Pandas dataframe with tokenized claims
    """

    # Check if 'claims' column exists in the dataframe
    if 'claims' not in df.columns:
        raise ValueError("The dataframe must contain a 'claims' column.")

    
    # Initialize gpt2 tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    



    # Tokenize the 'claims' column
    df['tokenized_claims'] = df['claims'].apply(
        lambda claims: [gpt2_tokenizer.tokenize(claim) for claim in claims])

    # Encode the tokenized content

    df['encoded_claims'] = df['tokenized_claims'].apply(
        lambda tokenized_content : [gpt2_tokenizer.encode_plus(content,
        return_tensors='tf', return_attention_mask=True) for content in tokenized_content]
    )




    # decoding the claims
    # df['decoded_claims'] = df['encoded_claims'].apply(lambda encoded_content: [gpt2_tokenizer.decode(encoded,skip_special_token=True) for encoded in encoded_content])


    



    # Write the 2 columns (tokenized and encoded) to an Excel file
    df.to_excel(output_file_name)

    return df


filter_ = filter_by_covid('factver1.xlsx', COVID_NAMES)




