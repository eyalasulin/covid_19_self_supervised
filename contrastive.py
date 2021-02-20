import tensorflow as tf
import numpy as np
import os
import statistics
import gc
from tensorflow.python.framework.ops import EagerTensor

from models import bmv_loss, create_contrastive_model


class batch_generator:
    def __init__(self, batch_size_):
        self.samples = []
        self.img_list = []
        self.luna_dir = './train/'
        self.batch_size = batch_size_
        self.data = []
        self.batch = []
        self.image_size = [160, 160]
        self.crop_size = [120, 120, 3]

    def create_img_list(self):
        for img in os.listdir(self.luna_dir):
            full_path = os.path.join(self.luna_dir, img)
            self.img_list.append(full_path)

    def load_luna(self, dev_mode):
        i = 0
        for img in self.img_list:
            i += 1
            if dev_mode:
                if i > 100:
                    break
            loaded_img = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(img))
            self.samples.append(loaded_img)
            # i need to use the real image and one augmented img and than add batch_size -2 augmented images
            # from different source images

    def gen_batch_2(self):
        num_of_batches = len(self.samples) // self.batch_size  # e.g 100//15 = 6
        epoch_mv = []

        for i in range(num_of_batches):  # e.g: 6
            batch_1 = []
            batch_2 = []
            for j in range(self.batch_size):  # 15
                batch_1.append(self.get_tensor_aug_img(self.samples[i]))
                batch_2.append(self.get_tensor_aug_img(self.samples[i]))
            mv_batch = [batch_1, batch_2]
            mv_batch = np.array(mv_batch)

            epoch_mv.append(mv_batch)

        return epoch_mv

    def get_tensor_aug_img(self, img):
        source_img = tf.image.resize(img, self.image_size)
        self.batch.append(tf.convert_to_tensor(source_img))
        aug_img = tf.image.random_flip_left_right(tf.image.random_crop(value=img, size=self.crop_size))
        aug_img = tf.image.resize(aug_img, self.image_size)
        return aug_img


def train_contrastive_layer(dev_mode=False,
                            batch_size=16,
                            pre_encode_size=1024,
                            projection_units=128,
                            EPOCHS=50,
                            model_name=f'contrastive_layers_err.h5',
                            freeze_base_model=False,
                            learning_rate=1e-3):  # can be also 40-45

    # Creating model
    self_trans_model = create_contrastive_model(pre_encode_size=pre_encode_size, projection_units=projection_units, inner_trainable=not freeze_base_model)

    # Training loop
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    print("Train started:")
    epoch_data = None
    for e in range(EPOCHS):
        # Generate batches
        if epoch_data is None:
            gen = batch_generator(batch_size)
            gen.create_img_list()
            gen.load_luna(dev_mode)
            epoch_data = gen.gen_batch_2()  # 1000//batch*batch*2*160*160*3
        epoch_loss = []
        print("starting epoch")
        #     epoch = gen.gen_batch_2()
        for idx in range(len(epoch_data)):
            with tf.GradientTape() as tape:
                logits_1 = self_trans_model(epoch_data[idx][0], training=True)
                logits_2 = self_trans_model(epoch_data[idx][1], training=True)
                mvb_loss = bmv_loss(logits_1, logits_2, 0.1)
                mvb_loss: EagerTensor
                grads = tape.gradient(mvb_loss, self_trans_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self_trans_model.trainable_weights))
            if not idx % 10:
                print(mvb_loss.numpy())

            epoch_loss.append(mvb_loss.numpy())
        avrg_loss = statistics.mean(epoch_loss)
        print(f"Epoch Num {e}, Avrg Loss: {avrg_loss}")
        print(gc.collect())
    print("success")

    self_trans_model.save(model_name)
    return self_trans_model


def test_save_load_contrastive():
    pass
