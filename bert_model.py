import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertModel

def getTokenizer():
    return BertTokenizer.from_pretrained("bert-base-cased")

def getToken(tokenizer, df):
    token = tokenizer.encode_plus(
        df['text'].iloc[0],
        max_length=256,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )
    return token

def createInputs(df):
    X_input_ids = np.zeros((len(df), 256))
    X_attn_masks = np.zeros((len(df), 256))
    return X_input_ids, X_attn_masks

def generate_training_data(df, ids, masks, tokenizer):
    for i, text in tqdm(enumerate(df['text'])):
        tokenized_text = tokenizer.encode_plus(
            text,
            max_length=256,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='tf'
        )
        ids[i, :] = tokenized_text.input_ids
        masks[i, :] = tokenized_text.attention_mask
    return ids, masks

def assignTrainingData(x_input_ids, x_attn_masks, df, tokenizer):
    X_input_ids, X_attn_masks = generate_training_data(df, x_input_ids, x_attn_masks, tokenizer)
    return X_input_ids, X_attn_masks

def initializeLabels(df, n):
    labels = np.zeros((len(df), n))
    labels[np.arange(len(df)), df['labels'].values] = 1
    return labels

def createDataset(x_input_ids, x_attn_masks, labels):
    dataset = tf.data.Dataset.from_tensor_slices((x_input_ids, x_attn_masks, labels))
    dataset.take(1)
    return dataset

def DatasetMapFunction(input_ids, attn_masks, labels):
    return {
        'input_ids': input_ids,
        'attention_mask': attn_masks
    }, labels

def mapDataset(dataset):
    dataset = dataset.map(DatasetMapFunction)
    dataset.take(1)
    return dataset

def setTrainSize(dataset, df):
    dataset = dataset.shuffle(10000).batch(16, drop_remainder=True)
    dataset.take(1)

    p = 0.8
    train_size = int((len(df) // 16) * p)
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    return train_dataset, val_dataset

def createBertModel(n):
    model = TFBertModel.from_pretrained('bert-base-cased')
    input_ids = tf.keras.layers.Input(shape=(256,), name='input_ids', dtype='int32')
    attn_masks = tf.keras.layers.Input(shape=(256,), name='attention_mask', dtype='int32')

    bert_embds = model.bert(input_ids, attention_mask=attn_masks)[
        1]  # 0 -> activation layer (3D), 1 -> pooled output layer (2D)
    intermediate_layer = tf.keras.layers.Dense(512, activation='relu', name='intermediate_layer')(bert_embds)
    output_layer = tf.keras.layers.Dense(n, activation='softmax', name='output_layer')(
        intermediate_layer)  # softmax -> calcs probs of classes

    out_model = tf.keras.Model(inputs=[input_ids, attn_masks], outputs=output_layer)
    return out_model

def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length=256,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
        'attention_mask': tf.cast(token.attention_mask, tf.float64)
    }

def make_prediction(model, processed_data, classes):
    probs = model.predict(processed_data)[0]
    return classes[np.argmax(probs)]

def getModel(df, no_of_classes, model_path):
    tokenizer = getTokenizer()
    token = getToken(tokenizer, df)
    X_input_ids, X_attn_masks = createInputs(df)
    X_input_ids, X_attn_masks = assignTrainingData(X_input_ids, X_attn_masks, df, tokenizer)
    labels = initializeLabels(df, no_of_classes)
    dataset = createDataset(X_input_ids, X_attn_masks, labels)
    dataset = mapDataset(dataset)
    train_dataset, val_dataset = setTrainSize(dataset, df)
    model = createBertModel(no_of_classes)
    model = tf.keras.models.load_model(model_path)
    return model, tokenizer
