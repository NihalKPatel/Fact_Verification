
import tensorflow as tf
import pandas as pd
from transformers import TFGPT2Model, TFBertModel
from annotate import COVID_NAMES
from transformer_model import transformed_data
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import logging

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import TFBertModel

from transformer_model import transformed_data


# Define a custom layer to extract BERT embeddings
class BertEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, bert_model, **kwargs):
        super(BertEmbeddingLayer, self).__init__(**kwargs)
        self.bert_model = bert_model

    def call(self, inputs):
        # Inputs should be a list of two tensors: input_ids and attention_mask
        input_ids, attention_mask = inputs
        embeddings = self.bert_model(input_ids, attention_mask=attention_mask).last_hidden_state
        return embeddings


def preparing_model(df, ):
    """
    Preparing input ids and mask attentions to feed to the GPT2 model

    Args:
        df (pandas.core.frame.DataFrame): _description_

    Returns:
        tuple: return list format of input ids and attention masks
    """

    df = df.copy()

    # Use .loc for assignments to avoid the warning
    df.loc[:, 'claim_input_ids'] = df['encoded_claims'].apply(lambda claim: claim['input_ids'])
    df.loc[:, 'claim_atten_mask'] = df['encoded_claims'].apply(lambda claim: claim['attention_mask'])
    df.loc[:, 'evidence_input_ids'] = df['encoded_evidences'].apply(lambda evidence: evidence['input_ids'])
    df.loc[:, 'evidence_atten_mask'] = df['encoded_evidences'].apply(lambda evidence: evidence['attention_mask'])
    # df ['reason_input_ids'] = df['encoded_reasons'].apply(lambda reason: reason['input_ids'])
    # df ['reason_atten_mask'] = df['encoded_reasons'].apply(lambda reason: reason['attention_mask'])

    df.to_excel('output_datasets/modelset_final.xlsx')

    return np.array(df['claim_input_ids'].tolist(), dtype=np.int32), np.array(df['claim_atten_mask'].tolist(),
                                                                              dtype=np.int32), np.array(
        df['evidence_input_ids'].tolist(), dtype=np.int32), np.array(df['evidence_atten_mask'].tolist(), dtype=np.int32)


def reshape_tensors(*args):
    return [tf.reshape(tensor, (-1, 512)) for tensor in args]


def claim_model(*args):

    """
    Claim Verification Model API, which will use GPT2 Model to output ...

    Args:
        input_ids (list): list of input ids
        atten_masks (list): list of attention masks
    """
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    tf.config.run_functions_eagerly(True)
    tf.compat.v1.keras.callbacks.ProgbarLogger(count_mode='steps')

    try:
        bert_model = TFBertModel.from_pretrained('bert-base-uncased',
                                                 trainable=True)  # will be used to generate embeddings, and as input to our model

        claim_input_ids = tf.keras.Input(shape=(512,), name="claim_input_ids", dtype=tf.int32)
        claim_attention_mask = tf.keras.Input(shape=(512,), name="claim_attention_mask", dtype=tf.int32)
        evidence_input_ids = tf.keras.Input(shape=(512,), name="evidence_input_ids", dtype=tf.int32)
        evidence_attention_mask = tf.keras.Input(shape=(512,), name="evidence_attention_mask", dtype=tf.int32)
        # reason_input_ids = tf.keras.Input(shape=(512,), name="reason_input_ids",dtype=tf.int32)
        # reason_attention_mask = tf.keras.Input(shape=(512,), name="reason_attention_mask",dtype=tf.int32)

        claim_inputs = {
            'input_ids': claim_input_ids,
            'attention_mask': claim_attention_mask
        }
        evidence_inputs = {
            'input_ids': evidence_input_ids,
            'attention_mask': evidence_attention_mask
        }
        '''
        reason_inputs = {
            'input_ids': reason_input_ids,
            'attention_mask': reason_attention_mask
        }
        '''

        # Create separate embedding layers for each input type
        bert_embedding_layer = BertEmbeddingLayer(bert_model, name="bert_embedding")
        claim_embedding = bert_embedding_layer([*claim_inputs.values()])
        evidence_embedding = bert_embedding_layer([*evidence_inputs.values()])
        # reason_embedding = bert_embedding_layer([*reason_inputs.values()])

        # Add any additional layers for processing the embeddings if needed
        # For example, you can concatenate the embeddings
        concatenated_embeddings = tf.keras.layers.Concatenate()([claim_embedding, evidence_embedding])
        concatenated_embeddings = tf.keras.layers.GlobalAveragePooling1D()(concatenated_embeddings)

        # Add your classification layers on top of the concatenated embeddings
        classification_output = tf.keras.layers.Dense(3, activation='softmax', name='veracity_layer')(
            concatenated_embeddings)

        model = tf.keras.Model(
            inputs=[claim_input_ids, claim_attention_mask, evidence_input_ids, evidence_attention_mask],
            outputs=classification_output)

        df = pd.read_excel("output_datasets/modelset_final.xlsx")
        # Strip any leading/trailing whitespace from the labels
        df['Label'] = df['Label'].str.strip()
        # Map the string labels to integers
        Label_mapping = {'T': 0, 'F': 1, 'N': 2}
        df['Label'] = df['Label'].map(Label_mapping)
        # Handle any missing or unmapped labels
        if df['Label'].isnull().any:
            print(f"Warning: {df['Label'].isnull().sum()} labels were not mapped correctly")
            df = df.dropna(subset=['Label'])  # Drop rows with missing labels
        # Convert labels to one-hot encoding
        tf_labels = tf.keras.utils.to_categorical(df['Label'].astype(int), num_classes=3)
        tf_labels = tf.reshape(tf_labels, (-1, 3))
        '''
        tf_labels = tf.constant([0,1,2],dtype = tf.int32)
        tf_labels = tf.one_hot(tf_labels, depth=3)
        '''

        shaped_tensors = reshape_tensors(*args)
        features_dataset = tf.data.Dataset.from_tensor_slices((
            (shaped_tensors[0], shaped_tensors[1]), (shaped_tensors[2], shaped_tensors[3])
        ))

        labels_dataset = tf.data.Dataset.from_tensor_slices((
            tf_labels
        ))

        dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))

        # calculate the total number of samples (rows)
        samples = len(args[0])
        # specify the training dataset size
        train_size = int(samples * 0.8)

        test_size = samples - train_size  # newly added line
        # shuffle the dataset
        dataset = dataset.shuffle(buffer_size=samples, seed=42, reshuffle_each_iteration=False)

        # split into train and test dataset
        print(f"Total samples: {samples}")
        print(f"Train size: {train_size}")
        print(f"Test size: {test_size}")
        train_dataset = dataset.skip(test_size).batch(1)
        test_dataset = dataset.take(test_size).batch(1)
        # Compile the classification model

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

        # Train the classification model
        validation_steps = len(test_dataset)
        print(f"size of test_dataset is {validation_steps}")
        model.fit(train_dataset, epochs=4, validation_data=test_dataset, validation_steps=validation_steps)

        # Test the model
        test_loss, test_acc = model.evaluate(test_dataset)

        prediction = model.predict(test_dataset)
        print(f"prediction /veracity: {prediction}")
        veracity_labels = np.argmax(prediction, axis=1)
        # Mapping from integer labels to veracity classes
        
        label_mapping = {0: 'T', 1: 'F', 2: 'N'}

        # Convert integer labels to veracity classes
        # veracity_labels = [label_mapping[label] for label in veracity_labels]
        print(veracity_labels)

        # true_labels = [Label_mapping[label] for label in df['Label']]  # Assuming 'Label' column contains ground
        # truth labels predicted_labels = [Label_mapping[label] for label in prediction]

        # recall = recall_score(true_labels, predicted_labels, average='weighted')

        # print(f"Label predicitions : {predicited_labels}")
        # print(f"recall : {recall}")

        print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')

        true_labels = df['Label'].values
        true_labels = true_labels[:test_size]
        precision = precision_score(true_labels, veracity_labels, average='weighted')
        recall = recall_score(true_labels, veracity_labels, average='weighted')
        f1 = f1_score(true_labels, veracity_labels, average='weighted')
        print(f'Precision: {precision}, Recall: {recall}, F1-Score: {f1}')

    except Exception as e:

        with open('log.txt', 'w') as log:
            log.write(str(e) + " " + str(sys.exc_info()[-1].tb_lineno))

logging.getLogger("tensorflow").setLevel(logging.ERROR)

claim_input_ids, claim_atten_masks, evidence_input_ids, evidence_atten_masks = preparing_model(transformed_data)

claim_model(claim_input_ids, claim_atten_masks, evidence_input_ids, evidence_atten_masks)

# TODO: 1. Use tensorflow
# train model
# test model
# deploy model

# TODO AFTER: #create a user friendly interface to use the model.

# DESCRIPTION: the model accept text and see if it is true or not, the model has a veracity algorithm that give a
# specific number of truthness

# DECODING HAPPEN AFTER DATA is trained.
