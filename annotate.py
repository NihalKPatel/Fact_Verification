import re

import pandas as pd

# Constants
COVID_NAMES = ["covid", "coronavirus"]



class DataAnnotation:
    '''
    Clean and label data and initializing the data in a new column
    '''
    def __init__(self,file_name,keywords):
        self.file_name = file_name
        self.keywords = keywords
        
        if not isinstance(keywords, list):
            raise ValueError("keywords param need to be list of strings (a list)")

        if not file_name.endswith(".xlsx") and not file_name.endswith(".csv"):
            raise ValueError("File must be a csv or xlsx.")
    def clean_data(self):
        data = pd.read_excel(f"datasets/{self.file_name}")  # Reading the dataset
        pd_filter = "|".join(self.keywords)
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
        ]  # filter the dataset by the given keywords,filter "out" incomplete data.

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
                if any(keyword for keyword in self.keywords if keyword in e.lower())
            ]
            for list_ in output["content"]
        ]  # adding the filtered claims in a new column "claims"

        output.to_excel(
            "output_datasets/factver1output.xlsx"
        )  # Write the output in a new file

        return output  # Return the output



filter_ = DataAnnotation("factver1.xlsx", COVID_NAMES)
filter_.clean_data()


# Function to filter data by COVID-related keywords
def filter_by_covid(file_name, keywords):
    """
    Clean data and initialize the data in a new column.
    Args:
        file_name (str): The file name to be read.
        keywords (list): List of strings to filter by.
    Returns:
        pd.DataFrame: Filtered Pandas DataFrame.
    """
    # Validate input
    if not isinstance(keywords, list):
        raise ValueError("Keywords must be a list of strings.")
    if not (file_name.endswith(".xlsx") or file_name.endswith(".csv")):
        raise ValueError("File must be a CSV or XLSX.")

    # Read data
    data = pd.read_excel(f"datasets/{file_name}")

    # Create filter pattern
    pd_filter = "|".join(keywords)

    # List of escape characters to remove
    escape_chars = list(map(re.escape, ['"', "*", "/", "(", ")", "\\n", "\\", "\\t", "\\u"]))

    # Filter and clean data
    output = data[
        data["content"].str.contains(pd_filter, case=False) & data["content"].str.endswith("]") |
        data["headline"].str.contains(pd_filter, na=False)
        ].copy()
    output.loc[:, "content"] = output["content"].replace(escape_chars, "", regex=True)
    output.loc[:, "content"] = output["content"].replace(r'[^\x00-\x7F]', "", regex=True)
    output.loc[:, "content"] = output["content"].apply(lambda x: x[1:-1].split(","))

    # Create a new column with filtered claims
    output.loc[:, "claims"] = [
        [e for e in list_ if any(keyword.lower() in e.lower() for keyword in keywords)]
        for list_ in output["content"]
    ]

    # Save to Excel
    output.to_excel("output_datasets/factver1output.xlsx")

    return output


# Function call
filter_by_covid("factVer1.3.xlsx", COVID_NAMES)
