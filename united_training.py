
from sklearn.metrics import roc_auc_score
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
import numpy as np
from contrastive import train_contrastive_layer
from models import Actions, create_united_model

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)
DB_UNITED_DIR = './Images-processed/'
CONTRASTIVE_BATCH_SIZE = 16


# creating covid dataset

def main(params):
    pre_encode_size = 1024
    projection_units = 128

    if params.action in [Actions.TrainWithContrastive.value, Actions.TrainWithTransfer.value]:
        train_dataset = image_dataset_from_directory(DB_UNITED_DIR, validation_split=0.2, subset="training", seed=123, image_size=(160, 160), batch_size=32)
        validation_dataset = image_dataset_from_directory(DB_UNITED_DIR, validation_split=0.2, subset="validation", seed=123, image_size=(160, 160),
                                                          batch_size=32)

        val_batches = tf.data.experimental.cardinality(validation_dataset)
        validation_dataset = validation_dataset.skip(val_batches // 5)

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
        # test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

        # create the base model
        if params.action == Actions.TrainWithContrastive.value:
            train_contrastive_layer(dev_mode=False,
                                    batch_size=CONTRASTIVE_BATCH_SIZE,
                                    pre_encode_size=1024,
                                    projection_units=128,
                                    EPOCHS=params.contrastive_epochs,
                                    model_name=params.contrastive_model_path,
                                    freeze_base_model=params.freeze_base_model,
                                    learning_rate=params.contrastive_learning_rate)
            united_model = create_united_model(pre_encode_size, projection_units, contrastive_path=params.contrastive_model_path, action=params.action)
        elif params.action == Actions.TrainWithTransfer.value:
            united_model = create_united_model(pre_encode_size, projection_units, contrastive_path=params.contrastive_model_path, action=params.action)
        else:
            raise NotImplementedError(params.action)

        history = united_model.fit(train_dataset,
                                   epochs=params.unite_epochs,
                                   validation_data=validation_dataset)

        x_val = np.concatenate([x for x, y in validation_dataset], axis=0)
        y_val = np.concatenate([y for x, y in validation_dataset], axis=0)
        y_pred = united_model.predict(x_val)
        auc = roc_auc_score(y_val, y_pred)
        loss, accuracy = united_model.evaluate(validation_dataset)
        print("final loss: {:.2f}".format(loss))
        print("final_accuracy: {:.2f}".format(accuracy))
        print("success")
        return loss, accuracy, history, auc

    else:
        raise ValueError(f'got: {params.action}')


if __name__ == '__main__':
    pass
    unite_epochs = 5
    contrastive_epochs = 1
    action = Actions.TrainWithContrastive.value
    contrastive_path = f'./contrastive_models/contrastive_{contrastive_epochs}.h5'
    freeze_base_model = False
    contrastive_learning_rate = 1e-8

