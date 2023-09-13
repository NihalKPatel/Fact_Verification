import tensorflow as tf
import pandas as pd
from transformers import TFGPT2Model
from annotate import filter_by_covid, COVID_NAMES
from transformer_model import transform_data


def write_sample_data() -> None:
    data = pd.DataFrame()
    old_data = pd.read_excel('output_datasets/finalver1_complete.xlsx')
    for i in ['headline','url','encoded_claims']:
        data[i] = old_data[i]
    data.to_excel('output_datasets/sample.xlsx')


def preparing_model(df,):
    """
    Preparing input ids and mask attentions to feed to the GPT2 model

    Args:
        df (pandas.core.frame.DataFrame): _description_

    Returns:
        tuple: return list format of input ids and attention masks
    """
    df['input_ids'] = df['encoded_claims'].apply(lambda claims:[claim['input_ids'] for claim in claims])
    df['atten_mask'] = df['encoded_claims'].apply(lambda claims:[claim['attention_mask'] for claim in claims])
    df.to_excel('output_datasets/finalver1_complete.xlsx')

    return df['input_ids'].tolist(),df['atten_mask'].tolist()
def claim_model(input_ids:list, atten_masks:list):
    """
    Claim Verification Model API, which will use GPT2 Model to output ...

    Args:
        input_ids (list): list of input ids
        atten_masks (list): list of attention masks
    """

    model = TFGPT2Model.from_pretrained('gpt2') # will be used to generate embeddings, and as input to our model


    # Add your classification layers on top of the GPT-2 embeddings
    classification_model = tf.keras.Sequential([
        model,  # GPT-2 model
        tf.keras.layers.Flatten(),
        # adding a layer for the features
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)), #input shape number corresponds to number of features
        # adding a layer for the labels (True, False, Invalid)
        tf.keras.layers.Dense(3, activation='softmax',name='veracity_layer')  # Binary classification output
])
    # Compile the classification model
    classification_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the classification model
    classification_model.fit([input_ids, attention_mask], labels, epochs=5, batch_size=32)






filter_ = filter_by_covid('factver1.xlsx', COVID_NAMES)
transform = transform_data(filter_)
input_ids, atten_masks = preparing_model(transform)
#claim_model(input_ids,atten_masks)
write_sample_data()





# TODO: 1. Use tensorflow
# train model
# test model
# deploy model

# TODO AFTER: #create a user friendly interface to use the model.

#DESCRIPTION: the model accept text and see if it is true or not, the model has a veracity algorithm that give a
# specific number of truthness

#DECODING HAPPEN AFTER DATA is trained.


