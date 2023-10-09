import re
import pandas as pd

COVID_NAMES = [
    "covid",
    "coronavirus",
]


class DataAnnotation:
    '''
    Clean data and initializing the data in a new column
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

