import tensorflow as tf
import pandas as pd
from transformers import TFGPT2Model
from annotate import filter_by_covid, COVID_NAMES
from transformer_model import transformed_data


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
    df['claim_input_ids'] = df['encoded_claims'].apply(lambda claim:claim['input_ids'])
    df['claim_atten_mask'] = df['encoded_claims'].apply(lambda claim:claim['attention_mask'])
    df ['evidence_input_ids'] = df['encoded_evidences'].apply(lambda evidence: evidence['input_ids'] )
    df ['evidence_atten_mask'] = df['encoded_evidences'].apply(lambda evidence: evidence['attention_mask'])
    df ['reason_input_ids'] = df['encoded_reasons'].apply(lambda reason: reason['input_ids'])
    df ['reason_atten_mask'] = df['encoded_reasons'].apply(lambda reason: reason['attention_mask'])


    df.to_excel('output_datasets/modelset_final.xlsx')

    return df['claim_input_ids'].tolist(),df['claim_atten_mask'].tolist(), df['evidence_input_ids'].tolist(),df['evidence_atten_mask'].tolist(), df['reason_input_ids'].tolist(),df['reason_atten_mask'].tolist()


def claim_model(*args):
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
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(3,)), #input shape number corresponds to number of features
        # in our case , the number of features is 3 : claim, evidence, and reason.
        # adding a layer for the labels (True, False, Invalid)
        tf.keras.layers.Dense(3, activation='softmax',name='veracity_layer')  # Binary classification output
])

    tf_labels = tf.constant([0,1,2],dtype = tf.int32)

    print(args[0])
    
    dataset = tf.data.Dataset.from_tensor_slices((
        (args[0],args[1]),(args[2],args[3]),(args[4],args[5])
   ,
    tf_labels  
))
    #calculate the total number of samples (rows)
    samples = len(args[0])
    # specify the training dataset size
    train_size = int(samples*0.7)
    #shuffle the dataset
    dataset = dataset.shuffle(buffer_size=samples, seed=42, reshuffle_each_iteration=False)

    #split into train and test dataset
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    # Compile the classification model
    classification_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the classification model
    classification_model.fit(train_dataset, epochs=5)
    
    # Test the model
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')


    



claim_input_ids, claim_atten_masks, evidence_input_ids, evidence_atten_masks,reason_input_ids, reason_atten_masks = preparing_model(transformed_data)

claim_model(claim_input_ids,claim_atten_masks,evidence_input_ids,evidence_atten_masks, reason_input_ids,reason_atten_masks)






# TODO: 1. Use tensorflow
# train model
# test model
# deploy model

# TODO AFTER: #create a user friendly interface to use the model.

#DESCRIPTION: the model accept text and see if it is true or not, the model has a veracity algorithm that give a
# specific number of truthness

#DECODING HAPPEN AFTER DATA is trained.


