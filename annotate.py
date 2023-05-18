import pandas as pd

COVID_NAMES = "|".join(
    [
        "covid",
        "covid 19",
        "Covid",
        "Covid 19",
        "covid-19",
        "Covid-19",
        "covid19",
        "Covid19",
        "COVID",
        "COVID 19",
        "COVID-19",
        "COVID19",
        "Coronavirus",
        "coronavirus",
        "Coronavirus",
    ]
)


def filter_by_covid(file_name, keywords):
    """
    Read some dataset by some filter, and write the newly
    filtered dataframe into excel file

    Args:
        file_name (file): The file name to be read
        keywords (str): list of strings to filter by

    Returns:
        pandas.core.frame.DataFrame: Pandas dataframe output
    """

    data = pd.read_excel(f"datasets/{file_name}")  # Reading the dataset
    output = data[
        data["content"].str.contains(keywords, na=False)
        | data["headline"].str.contains(keywords, na=False)
    ]  # filter the dataset by the given keywords
    output.to_excel(
        "output_datasets/factver1output.xlsx"
    )  # Write the output in a new file
    return output  # Return the output


# filter_by_covid('factver1.xlsx',COVID_NAMES)
