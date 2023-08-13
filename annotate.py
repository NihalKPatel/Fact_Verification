import re
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer

COVID_NAMES = [
    "covid",
    "coronavirus",
]


def filter_by_covid(file_name, keywords):
    """
    Read some dataset by some filter, and write the newly
    filtered dataframe into excel file

    Args:
        file_name (file): The file name to be read
        keywords (list): list of strings to filter by

    Returns:
        pandas.core.frame.DataFrame: Pandas dataframe output
    """

    # Handling the passed parameters data

    if not isinstance(keywords, list):
        raise ValueError("keywords param need to be list of strings (a list)")

    if not file_name.endswith(".xlsx") and not file_name.endswith(".csv"):
        raise ValueError("File must be a csv or xlsx.")

    data = pd.read_excel(f"datasets/{file_name}")  # Reading the dataset
    pd_filter = "|".join(keywords)
    es_list = [
        '"',
        "*",
        "/",
        "(",
        ")",
        "\\n",
        "\\",
        "\\t",
        "\\u",
    ]  # to remove any escape characters and special characters
    escape_chars = list(map(re.escape, es_list))

    output = data[
        data["content"].str.contains(pd_filter, case=False)
        & data["content"].str.endswith("]")
        | data["headline"].str.contains(pd_filter, na=False)
        ].copy()

    output["content"] = output["content"].replace(
        escape_chars, "", regex=True
    )  # removing all escape sequences and special characters
    output["content"] = output["content"].replace(
        r'[^\x00-\x7F]', "", regex=True
    )  # remove all non-ASCII characters

    output.loc[:, "content"] = output["content"].apply(
        lambda x: x[1:-1].split(",")
    )  # convert the strings to list of strings

    output.loc[:, "claims"] = [
        [
            e
            for e in list_
            if any(keyword for keyword in keywords if keyword in e.lower())
        ]
        for list_ in output["content"]
    ]  # adding the filtered claims in a new column "claims"

    output.to_excel(
        "output_datasets/factver1output.xlsx"
    )  # Write the output in a new file

    return output  # Return the output


def tokenize_claims_with_bert(df, output_file_name="output_datasets/tokenized_claims_with_bert.xlsx"):
    """
    Tokenize the claims in a dataframe using BERT's Tokenizer

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

    # Tokenize the 'claims' column
    df['tokenized_claims'] = df['claims'].apply(
        lambda claims: [tokenizer.tokenize(claim) for claim in claims])

    # Write the tokenized claims to an Excel file
    df.to_excel(output_file_name)

    return df


# Example usage
filtered_data = filter_by_covid("FactVer1.3.xlsx", COVID_NAMES)

tokenize_claims_with_bert(filtered_data, "output_datasets/my_tokenized_claims_with_bert.xlsx")