from enum import Enum, auto

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.keras.engine.functional import Functional


class Actions(Enum):
    TrainWithTransfer = auto()
    TrainContrastiveLayer = auto()
    TrainWithContrastive = auto()


def create_contrastive_model(pre_encode_size, projection_units, inner_trainable=False):
    # Input size is images size
    input_size = (160, 160, 3)
    # projection units size was suggested by the paper author

    base_model = tf.keras.applications.MobileNetV2(input_shape=input_size,
                                                   include_top=False,
                                                   weights='imagenet')
    # freeze the base model
    base_model.trainable = inner_trainable
    inputs = keras.Input(shape=input_size)
    norm_input = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)(inputs)
    features = base_model(norm_input)
    flat = layers.Flatten()(features)
    pre_encode = layers.Dense(pre_encode_size)(flat)
    norm_encode = tf.math.l2_normalize(pre_encode)
    encode = layers.Dense(projection_units, activation="relu")(norm_encode)
    outputs = tf.math.l2_normalize(encode)
    model = keras.Model(inputs=inputs, outputs=outputs, name="contrastive")
    return model


def create_united_model(pre_encode_size, projection_units, contrastive_path, action):
    base_learning_rate = 0.0001
    input_size = (160, 160, 3)

    inputs = tf.keras.Input(shape=input_size)
    if action == Actions.TrainWithContrastive.value:
        contrastive_model = tf.keras.models.load_model(contrastive_path)
        contrastive_model.layers[2].trainable = False

    elif action == Actions.TrainWithTransfer.value:
        contrastive_model = create_contrastive_model(pre_encode_size, projection_units)
        contrastive_model.layers[2].trainable = False
    else:
        raise ValueError(f'got action = {action}')
    prediction_layer = tf.keras.layers.Dense(1)
    x = contrastive_model(inputs, training=True)  # pretrained mobilenet layers without classification or input layer
    outputs = prediction_layer(x)  # output logit

    united_model = tf.keras.Model(inputs, outputs, name="united_model")

    united_model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                         loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                         metrics=['accuracy'])
    print_summary = True
    if print_summary:
        print(f'len(united_model.trainable_variables): {len(united_model.trainable_variables)}')
        print('united_summary:')
        print(united_model.summary())
        print('contrastive_summary:')
        print(contrastive_model.summary())
        for layer in united_model.layers:
            print(layer.trainable, layer, type(layer))
            if type(layer) == Functional:
                for inner_layer in layer.layers:
                    print(inner_layer.trainable, inner_layer, type(inner_layer))
    return united_model


def print_model_info(model):
    print(f'len(united_model.trainable_variables): {len(model.trainable_variables)}')
    print('model_summary:')
    print(model.summary())
    for layer in model.layers:
        print(layer.trainable, layer, type(layer))
        if type(layer) == Functional:
            print('found functional layer:')
            for inner_layer in layer.layers:
                print(inner_layer.trainable, inner_layer, type(inner_layer))
            print('functional layer_summary:')
            print(layer.summary())
            print('emd details of functional layer.')


def bmv_loss(v1_batch, v2_batch, tau) -> EagerTensor:
    loss_multi_view_total = 0
    for i in range(len(v1_batch)):
        loss_i_positive = tf.math.exp(tf.tensordot(v1_batch[i], v2_batch[i], 1) / tau)
        loss_i_negative = 0
        for j in range(len(v1_batch)):
            if j != i:
                loss_i_negative += tf.math.exp(tf.tensordot(v1_batch[i], v2_batch[j], 1) / tau)
        loss_i = loss_i_positive / loss_i_negative
        loss_i = - tf.math.log(loss_i)
        loss_multi_view_total += loss_i
    return loss_multi_view_total


def prepare_contrastive():
    pass
