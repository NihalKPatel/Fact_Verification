from transformers import BertTokenizer, GPT2Tokenizer, GPT2LMHeadModel

from annotate import filter_by_covid, COVID_NAMES


# Function to tokenize, encode, and decode data
def transform_data(df, output_file_name="output_datasets/finalver1_complete.xlsx"):
    """
    Tokenize, Encode, and Decode data.
    Args:
        df (pd.DataFrame): DataFrame to be tokenized.
        output_file_name (str): Output file name.
    Returns:
        pd.DataFrame: DataFrame with tokenized claims.
    """
    # Check for 'claims' column
    if 'claims' not in df.columns:
        raise ValueError("The DataFrame must contain a 'claims' column.")

    # Initialize tokenizers and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Tokenize claims
    df['tokenized_claims'] = df['claims'].apply(lambda claims: [tokenizer.tokenize(claim) for claim in claims])

    # Encode tokenized claims
    df['encoded_claims'] = df['tokenized_claims'].apply(
        lambda tokens: [gpt2_tokenizer.encode(token) for token in tokens])

    # Decode encoded claims
    df['decoded_claims'] = df['encoded_claims'].apply(
        lambda encodings: [gpt2_tokenizer.decode(encoding, skip_special_tokens=True) for encoding in encodings])

    '''
    # Embed the tokenized contents:
    df['embedded_claims'] = df['encoded_claims'].apply(
        lambda ids: [model(id_).last_hidden_state for id_ in ids]
    )
    '''

    # Save to Excel
    df.to_excel(output_file_name)

    return df


# Function call
filtered_data = filter_by_covid('factVer1.3.xlsx', COVID_NAMES)
transform_data(filtered_data)
