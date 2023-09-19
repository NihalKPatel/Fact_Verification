from transformers import GPT2Tokenizer
import pandas as pd
import tensorflow as tf
from annotate import filter_by_covid, COVID_NAMES


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


def convert_pandas(file: str):
    df = pd.read_excel(file)
    return df


def transform_data(df, output_file_name):
    global gpt2_tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Clear columns from NAN and float values
    df.fillna('', inplace=True)

    # Tokenize the 'claims' column if it exists
    if 'claims' in df.columns:
        df['tokenized_claims'] = df['claims'].apply(lambda claims: [gpt2_tokenizer.tokenize(claim) for claim in claims])
        df['encoded_claims'] = df['tokenized_claims'].apply(encode_and_pad)

    # Tokenize the 'Claim', 'Evidence', and 'Reason' columns if they exist
    if 'Claim' in df.columns and 'Evidence' in df.columns and 'Reason' in df.columns:
        df['tokenized_claims'] = df['Claim'].apply(lambda claim: gpt2_tokenizer.tokenize(claim))
        df['tokenized_evidences'] = df['Evidence'].apply(lambda evidence: gpt2_tokenizer.tokenize(evidence))
        df['tokenized_reasons'] = df['Reason'].apply(lambda reason: gpt2_tokenizer.tokenize(reason))

        # Encode the tokenized content
        df['encoded_claims'] = df['tokenized_claims'].apply(
            lambda tokenized_content: gpt2_tokenizer.encode_plus(tokenized_content, return_tensors='tf',
                                                                 return_attention_mask=True, max_length=100,
                                                                 truncation=True, padding=True))
        df['encoded_evidences'] = df['tokenized_evidences'].apply(
            lambda tokenized_content: gpt2_tokenizer.encode_plus(tokenized_content, return_tensors='tf',
                                                                 return_attention_mask=True, max_length=100,
                                                                 truncation=True, padding=True))
        df['encoded_reasons'] = df['tokenized_reasons'].apply(
            lambda tokenized_content: gpt2_tokenizer.encode_plus(tokenized_content, return_tensors='tf',
                                                                 return_attention_mask=True, max_length=100,
                                                                 truncation=True, padding=True))

    df.to_excel(output_file_name)
    return df


# Using the functions
filter_ = filter_by_covid('factver1.xlsx', COVID_NAMES)
transformed_df = transform_data(filter_, output_file_name="output_datasets/finalver1_complete.xlsx")

pd_obj = convert_pandas('datasets/annotated_data.xlsx')
transformed_data = transform_data(pd_obj, 'output_datasets/modelset.xlsx')