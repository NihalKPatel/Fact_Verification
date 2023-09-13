from transformers import GPT2Tokenizer
from annotate import filter_by_covid, COVID_NAMES
import tensorflow as tf


def encode_and_pad(tokenized_content):
    max_length = 512
    input_ids_list = []
    attention_mask_list = []

    for content in tokenized_content:
        encoded = gpt2_tokenizer.encode_plus(content, return_attention_mask=True, padding='max_length',
                                             max_length=max_length, truncation=True)
        input_ids_list.append(encoded['input_ids'])
        attention_mask_list.append(encoded['attention_mask'])

    input_ids_tensor = tf.stack(input_ids_list)
    attention_mask_tensor = tf.stack(attention_mask_list)

    return {'input_ids': input_ids_tensor, 'attention_mask': attention_mask_tensor}


def transform_data(df, output_file_name="output_datasets/finalver1_complete.xlsx"):
    """
    Tokenize, Encode, and Decode data.

    Args:
        df (pandas.core.frame.DataFrame): Pandas dataframe to be tokenized
        output_file_name (str): The file name to write the tokenized claims

    Returns:
        pandas.core.frame.DataFrame: Pandas dataframe with tokenized claims
    """
    if 'claims' not in df.columns:
        raise ValueError("The dataframe must contain a 'claims' column.")

    global gpt2_tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    df['tokenized_claims'] = df['claims'].apply(lambda claims: [gpt2_tokenizer.tokenize(claim) for claim in claims])
    df['encoded_claims'] = df['tokenized_claims'].apply(encode_and_pad)

    df.to_excel(output_file_name)
    return df


filter_ = filter_by_covid('factver1.xlsx', COVID_NAMES)
transformed_df = transform_data(filter_, output_file_name="output_datasets/finalver1_complete.xlsx")
