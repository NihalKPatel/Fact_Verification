import pandas as pd
import tensorflow as tf
from transformers import TFGPT2Model
from transformer_model import transformed_data


def write_sample_data() -> None:
    data = pd.DataFrame()
    old_data = pd.read_excel('output_datasets/finalver1_complete.xlsx')
    for i in ['headline', 'url', 'encoded_claims']:
        data[i] = old_data[i]
    data.to_excel('output_datasets/sample.xlsx')


def preparing_model(df):
    def extract_input_ids_and_masks(encoded_column):
        input_ids = []
        attention_masks = []
        for encoded in encoded_column:
            input_ids.append(encoded['input_ids'].numpy().tolist())
            attention_masks.append(encoded['attention_mask'].numpy().tolist())
        return input_ids, attention_masks

    claim_input_ids, claim_atten_masks = extract_input_ids_and_masks(df['encoded_claims'])
    evidence_input_ids, evidence_atten_masks = extract_input_ids_and_masks(df['encoded_evidences'])
    reason_input_ids, reason_atten_masks = extract_input_ids_and_masks(df['encoded_reasons'])

    return claim_input_ids, claim_atten_masks, evidence_input_ids, evidence_atten_masks, reason_input_ids, reason_atten_masks


def claim_model(*args):
    """
    Claim Verification Model API, which will use GPT2 Model to output ...

    Args:
        input_ids (list): list of input ids
        atten_masks (list): list of attention masks
    """

    gpt_model = TFGPT2Model.from_pretrained('gpt2')  # will be used to generate embeddings, and as input to our model

    claim_input_ids = args[0]
    claim_attention_mask = args[1]

    evidence_input_ids = args[2]
    evidence_attention_mask = args[3]

    reason_input_ids = args[4]
    reason_attention_mask = args[5]

    claim_inputs = {
        'input_ids': claim_input_ids,
        'attention_mask': claim_attention_mask
    }
    evidence_inputs = {
        'input_ids': evidence_input_ids,
        'attention_mask': evidence_attention_mask
    }
    reason_inputs = {
        'input_ids': reason_input_ids,
        'attention_mask': reason_attention_mask
    }

    # Create separate embedding layers for each input type
    claim_embedding = gpt_model([claim_inputs]).last_hidden_state
    evidence_embedding = gpt_model([evidence_inputs]).last_hidden_state
    reason_embedding = gpt_model([reason_inputs]).last_hidden_state

    claim_embedding = tf.keras.layers.Flatten()(claim_embedding)
    evidence_embedding = tf.keras.layers.Flatten()(evidence_embedding)
    reason_embedding = tf.keras.layers.Flatten()(reason_embedding)

    # Add any additional layers for processing the embeddings if needed
    # For example, you can concatenate the embeddings
    concatenated_embeddings = tf.keras.layers.Concatenate()([claim_embedding, evidence_embedding, reason_embedding])

    # Add your classification layers on top of the concatenated embeddings
    classification_output = tf.keras.layers.Dense(10, activation=tf.nn.relu)(concatenated_embeddings)
    classification_output = tf.keras.layers.Dense(3, activation='softmax', name='veracity_layer')(classification_output)

    model = tf.keras.Model(
        inputs=[claim_input_ids, claim_attention_mask, evidence_input_ids, evidence_attention_mask, reason_input_ids,
                reason_attention_mask], outputs=classification_output)

    '''
    # Add your classification layers on top of the GPT-2 embeddings
    classification_model = tf.keras.Sequential([
        #model,  # GPT-2 model
        # adding a layer for the features
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(3,)), #input shape number corresponds to number of features
        # in our case , the number of features is 3 : claim, evidence, and reason.
        # adding a layer for the labels (True, False, Invalid)
        tf.keras.layers.Dense(3, activation='softmax',name='veracity_layer')  # Binary classification output
])
  '''

    tf_labels = tf.constant([0, 1, 2], dtype=tf.int32)
    tf_labels = tf.one_hot(tf_labels, depth=3)

    features_dataset = tf.data.Dataset.from_tensor_slices((
        (args[0], args[1]), (args[2], args[3]), (args[4], args[5])

    ))

    labels_dataset = tf.data.Dataset.from_tensor_slices((
        tf_labels
    ))

    dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))

    # calculate the total number of samples (rows)
    samples = len(args[0])
    # specify the training dataset size
    train_size = int(samples * 0.7)
    # shuffle the dataset
    dataset = dataset.shuffle(buffer_size=samples, seed=42, reshuffle_each_iteration=False)

    # split into train and test dataset
    train_dataset = dataset.take(train_size)

    test_dataset = dataset.skip(train_size)

    # Compile the classification model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the classification model
    validation_steps = len(test_dataset)
    model.fit(train_dataset, epochs=5, validation_data=test_dataset, validation_steps=validation_steps)

    # Test the model
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')


claim_input_ids, claim_attention_masks, evidence_input_ids, evidence_attention_masks, reason_input_ids, \
    reason_attention_masks = preparing_model(transformed_data)

claim_model(claim_input_ids, claim_attention_masks, evidence_input_ids, evidence_attention_masks, reason_input_ids,
            reason_attention_masks)

# TODO: 1. Use tensorflow
# train model
# test model
# deploy model

# TODO AFTER: #create a user friendly interface to use the model.

# DESCRIPTION: the model accept text and see if it is true or not, the model has a veracity algorithm that give a
# specific number of truthness

# DECODING HAPPEN AFTER DATA is trained.
