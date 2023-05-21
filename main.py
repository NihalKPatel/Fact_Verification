import nltk
import pandas as pd
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


def filter_covid_articles(df):
    # Filter for articles related to COVID-19
    df_covid = df[df['headline'].str.contains('covid|covid-19|vaccine', case=False)
                  | df['content'].str.contains('covid|covid-19|vaccine', case=False)]

    # Save the filtered dataset
    df_covid.to_csv('dataset/FactVer1.3_covid.csv', index=False, encoding='utf-8')


def filter_covid_claims_from_file(file_path):
    # Read the filtered dataset
    df_covid = pd.read_csv(file_path)

    # Combine 'headline' and 'content' columns into a single text column
    df_covid['text'] = df_covid['headline'].fillna('') + ' ' + df_covid['content'].fillna('')

    # Extract claims from each row
    claims = []
    for text in df_covid['text']:
        claims.extend(filter_covid_claims(text))

    # Save each claim in its own row
    df_claims = pd.DataFrame({'claim': claims})
    df_claims.to_csv('dataset/FactVer1.3_covid_claims.csv', index=False, encoding='utf-8')


def filter_covid_claims(text):
    # Split the text into separate claims based on the numbered labels
    claims = re.split(r'\d+:', text)

    # Define the COVID-related keywords
    keywords = ['covid', 'covid-19', 'vaccine']

    # Filter the claims based on the keywords
    covid_claims = [claim.strip() for claim in claims if any(keyword in claim.lower() for keyword in keywords)]

    return covid_claims


def main():
    # Load the dataset
    df = pd.read_excel('dataset/FactVer1.3.xlsx')

    filter_covid_articles(df)

    filter_covid_claims_from_file('dataset/FactVer1.3_covid.csv')


if __name__ == "__main__":
    main()
