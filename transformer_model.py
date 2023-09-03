from transformers import BertConfig, BertModel, BertTokenizer, GPT2Tokenizer, GPT2LMHeadModel
from annotate import filter_by_covid, COVID_NAMES
import torch
import pandas as pd




def transform_data(df, output_file_name="output_datasets/tokenized_claims_with_bert.xlsx"):
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

    # Initialize the BERT Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Initialize gpt2 tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Initialize GPT model for decoding
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    



    # Tokenize the 'claims' column
    df['tokenized_claims'] = df['claims'].apply(
        lambda claims: [tokenizer.tokenize(claim) for claim in claims])

    # Encode the tokenized content

    df['encoded_claims'] = df['tokenized_claims'].apply(
        lambda tokenized_content : [gpt2_tokenizer.encode(content) for content in tokenized_content]
    )


    # decoding the claims
    df['decoded_claims'] = df['encoded_claims'].apply(lambda encoded_content: [gpt2_tokenizer.decode(encoded,skip_special_token=True) for encoded in encoded_content])


    


     # Embed the tokenized contents:

    '''
    df['embedded_claims'] = df['encoded_claims'].apply(
        lambda ids: [model(id_).last_hidden_state for id_ in ids]
    )
    '''
    # Write the tokenized claims to an Excel file
    df.to_excel(output_file_name)

    return df


filter_ = filter_by_covid('factver1.xlsx', COVID_NAMES)

transform_data(filter_)


