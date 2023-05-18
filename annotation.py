import pandas as pd
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load the data
df = pd.read_csv('dataset/FactVer1.3.csv')

# Open a new file to write
with open('output.csv', 'w') as f:
    for index, row in df.iterrows():
        text = row['article_id']

        # Tokenization
        tokens = nltk.word_tokenize(text)
        f.write("Tokens:\n")
        f.write(str(tokens) + "\n")

        # POS Tagging
        pos_tags = nltk.pos_tag(tokens)
        f.write("\nPOS tags:\n")
        f.write(str(pos_tags) + "\n")

        # Named Entity Recognition
        named_entities = nltk.ne_chunk(pos_tags, binary=False)
        f.write("\nNamed entities:\n")
        f.write(str(named_entities) + "\n")