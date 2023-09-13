import pandas as pd
from transformers import GPT2Tokenizer

from annotate import filter_by_covid, COVID_NAMES


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
        lambda tokenized_content: [gpt2_tokenizer.encode_plus(content,
                                                              return_tensors='tf', return_attention_mask=True) for
                                   content in tokenized_content]
    )

    # decoding the claims df['decoded_claims'] = df['encoded_claims'].apply(lambda encoded_content: [
    # gpt2_tokenizer.decode(encoded,skip_special_token=True) for encoded in encoded_content])

    # Write the 2 columns (tokenized and encoded) to an Excel file
    df.to_excel(output_file_name)

    return df


# Tokenizing, Encoding, and Decoding function
def tokenize_claims_and_evidence(df, output_file_name="output_datasets/tokenized_claims_and_evidence.xlsx"):
    df = df.copy()  # To avoid SettingWithCopyWarning
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Tokenizing and Encoding 'Claim' column
    if 'Claim' in df.columns:
        df = df[df['Claim'].apply(lambda x: isinstance(x, str))]
        df['tokenized_claims'] = df['Claim'].apply(gpt2_tokenizer.tokenize)
        df['encoded_claims'] = df['tokenized_claims'].apply(
            lambda x: gpt2_tokenizer.encode_plus(x, return_tensors='tf', return_attention_mask=True)
        )

    # Tokenizing and Encoding 'Reason' column
    if 'Reason' in df.columns:
        df = df[df['Reason'].apply(lambda x: isinstance(x, str))]
        df['tokenized_reason'] = df['Reason'].apply(gpt2_tokenizer.tokenize)
        df['encoded_reason'] = df['tokenized_reason'].apply(
            lambda x: gpt2_tokenizer.encode_plus(x, return_tensors='tf', return_attention_mask=True)
        )

    # Tokenizing and Encoding 'Evidence' column
    if 'Evidence' in df.columns:
        df = df[df['Evidence'].apply(lambda x: isinstance(x, str))]
        df['tokenized_evidence'] = df['Evidence'].apply(gpt2_tokenizer.tokenize)
        df['encoded_evidence'] = df['tokenized_evidence'].apply(
            lambda x: gpt2_tokenizer.encode_plus(x, return_tensors='tf', return_attention_mask=True)
        )

    df.to_excel(output_file_name, index=False)
    return df


df_claims = pd.read_excel('Evidence/Claim and evidence.xlsx')  # Load the data

filter_ = filter_by_covid('factVer1.3.xlsx', COVID_NAMES)
tokenize_claims_and_evidence(df_claims)
transform_data(filter_)
