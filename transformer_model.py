from transformers import GPT2Tokenizer
from annotate import filter_by_covid, COVID_NAMES
import torch
import pandas as pd


def convert_pandas(file:str):
    """
    Covert excel sheet to pandas object

    Args:
        file (str): excel file path
    """

    df = pd.read_excel(file)
    #df.dropna(subset=['Claim','Reason'],how='all',inplace=True)
    return df


def transform_data(df, output_file_name):
    """
    Tokenize, Encode,  and Decode data.

    Args:
        df (pandas.core.frame.DataFrame): Pandas dataframe to be tokenized
        output_file_name (str): The file name to write the tokenized claims

    Returns:
        pandas.core.frame.DataFrame: Pandas dataframe with tokenized claims
    """


    
    # Initialize gpt2 tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_tokenizer.add_special_tokens({"pad_token": "[PAD]"})


    # clear columns from NAN and float values
    df['Claim'].fillna('', inplace=True)
    df['Evidence'].fillna('', inplace=True)
    df['Reason'].fillna('', inplace=True)

    # Tokenize the 'claims' column
    df['tokenized_claims'] = df['Claim'].apply(
        lambda claim: gpt2_tokenizer.tokenize(claim))

    # Filter out rows with empty tokenized claims
    df = df[df['tokenized_claims'].apply(lambda x: len(x) > 0)]

    # Tokenize the 'evidence column'
    df['tokenized_evidences'] = df['Evidence'].apply(
        lambda evidence: gpt2_tokenizer.tokenize(evidence)
    )

    df = df[df['tokenized_evidences'].apply(lambda x: len(x) > 0)]

    # Tokenize the 'reason column'
    df['tokenized_reasons'] = df['Reason'].apply(
        lambda reason: gpt2_tokenizer.tokenize(reason)
    )
    df = df[df['tokenized_reasons'].apply(lambda x: len(x) > 0)]


    # Encode the tokenized content

    df['encoded_claims'] = df['tokenized_claims'].apply(
        lambda tokenized_content : gpt2_tokenizer.encode_plus(tokenized_content,
        return_tensors='tf', return_attention_mask=True,max_length=512, truncation=True,padding='max_length')
    )

    df['encoded_evidences'] = df['tokenized_evidences'].apply(
        lambda tokenized_content : gpt2_tokenizer.encode_plus(tokenized_content,
        return_tensors='tf', return_attention_mask=True,max_length=512, truncation=True,padding='max_length')
    )

    df['encoded_reasons'] = df['tokenized_reasons'].apply(
        lambda tokenized_content : gpt2_tokenizer.encode_plus(tokenized_content,
        return_tensors='tf', return_attention_mask=True,max_length=512,truncation=True,padding='max_length')
    )







    # decoding the claims
    # df['decoded_claims'] = df['encoded_claims'].apply(lambda encoded_content: [gpt2_tokenizer.decode(encoded,skip_special_token=True) for encoded in encoded_content])


    



    # Write the 2 columns (tokenized and encoded) to an Excel file
    df.to_excel(output_file_name)

    return df


pd_obj = convert_pandas('datasets/annotated_data.xlsx') 

transformed_data = transform_data(pd_obj, 'output_datasets/modelset.xlsx')


