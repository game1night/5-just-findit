#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/5/8 14:50

@author: tatatingting
"""

# 1. Import the dataset
# 2. Explore the data
# 3. Preprocess the data
# 4. Build the model
# - Set up the layers
# - Compile the model
# 5. Train the model
# - Feed the model
# - Evaluate accuracy
# - Make predictions
# - Verify predictions
# 6. Use the trained model

#
# https://www.nvidia.cn
# https://tensorflow.google.cn/tutorials
#


import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random

AUTOTUNE = tf.data.experimental.AUTOTUNE


def build_model_batch(url):
    w = 216
    h = 216
    BATCH_SIZE = 32
    split_p = 0.33

    path_dir = os.path.dirname(__file__)
    path_img = os.path.join(path_dir, url)
    all_image_paths = os.listdir(path_img)
    random.shuffle(all_image_paths)

    # 确定每张图片的标签
    all_image_labels = []
    for file in all_image_paths:
        all_image_labels.append(np.int(file[0]))
    # print("First 10 labels indices: ", all_image_labels[:10])
    label_names = ['no', 'yes']
    all_image_paths = [os.path.join(path_img, path) for path in all_image_paths]
    # print("First 10 paths indices: ", all_image_paths[:10])

    # 构建一个 tf.data.Dataset
    # path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    # image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    # label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    # image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    train_image_paths, test_image_path, train_labels, test_labels = train_test_split(
        all_image_paths,
        all_image_labels,
        test_size=split_p,
        random_state=42,
    )

    plt.plot(train_labels)
    plt.show()
    plt.plot(test_labels)
    plt.show()

    image_count = len(train_image_paths)

    def preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [w, h])
        # normalize to [0,1] range
        image /= 255.0
        return image

    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        return preprocess_image(image)

    # 元组被解压缩到映射函数的位置参数中
    def load_and_preprocess_from_path_label(path, label):
        return load_and_preprocess_image(path), label

    ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))

    image_label_ds = ds.map(load_and_preprocess_from_path_label)

    # 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
    # 被充分打乱。
    ds = image_label_ds.shuffle(buffer_size=image_count)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    # ds = image_label_ds.apply(
    #     tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
    # ds = ds.batch(BATCH_SIZE)
    # ds = ds.prefetch(buffer_size=AUTOTUNE)

    # 1 mobile_net
    # mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
    # mobile_net.trainable = False
    # def change_range(image, label):
    #     return 2 * image - 1, label
    # keras_ds = ds.map(change_range)
    # 数据集可能需要几秒来启动，因为要填满其随机缓冲区。
    # image_batch, label_batch = next(iter(keras_ds))
    # feature_map_batch = mobile_net(image_batch)
    # model = tf.keras.Sequential([
    #     mobile_net,
    #     tf.keras.layers.GlobalAveragePooling2D(),
    #     tf.keras.layers.Dense(len(label_names), activation='softmax')])
    # logit_batch = model(image_batch).numpy()
    # model.compile(optimizer=tf.keras.optimizers.Adam(),
    #               loss='sparse_categorical_crossentropy',
    #               metrics=["accuracy"])

    # 2
    image_batch, label_batch = next(iter(ds))
    # print(image_batch.shape)
    model = tf.keras.Sequential([
        keras.layers.Flatten(input_shape=(w, h, 3)),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(len(label_names), activation='softmax')
    ])
    logit_batch = model(image_batch).numpy()
    # print("min logit:", logit_batch.min())
    # print("max logit:", logit_batch.max())
    # print()
    # print("Shape:", logit_batch.shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # begin
    # print(len(model.trainable_variables))
    # print(model.summary())
    steps_per_epoch = tf.math.ceil(image_count / BATCH_SIZE).numpy()
    # print(steps_per_epoch)

    # model.fit(ds, epochs=10, steps_per_epoch=steps_per_epoch, validation_data=ds, validation_steps=steps_per_epoch)
    model.fit(ds, epochs=10, steps_per_epoch=steps_per_epoch)

    # - Evaluate accuracy
    def get_ds(all_image_paths, all_image_labels, BATCH_SIZE):
        image_count = len(all_image_paths)

        ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
        image_label_ds = ds.map(load_and_preprocess_from_path_label)

        # 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
        # 被充分打乱。
        ds = image_label_ds.shuffle(buffer_size=image_count)
        ds = ds.repeat()
        ds = ds.batch(BATCH_SIZE)
        # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        steps_per_epoch = tf.math.ceil(image_count / BATCH_SIZE).numpy()

        return ds, steps_per_epoch

    ds_test, steps_count = get_ds(test_image_path, test_labels, BATCH_SIZE)

    test_loss, test_acc = model.evaluate(ds_test, verbose=2, steps=steps_count)
    print('\nTest accuracy:', test_acc)

    # 将整个模型保存为HDF5文件，非常大的！
    # !pip install -q pyyaml h5py  # 需要以 HDF5 格式保存模型
    model.save(
        os.path.join(os.path.dirname(__file__), 'new_model_{}_{}.h5'.format(np.around(test_acc, 4), time.time())))

    return None


def build_model_all(url):
    # Basic classification: Classify the splitted images
    print(tf.version.VERSION)

    # Import the dataset
    # 准备信息地址等
    path_dir = os.path.dirname(__file__)
    path_img = os.path.join(path_dir, url)
    imgs = []
    labels = []
    n = 0
    # 输出所有文件
    all_image_paths = os.listdir(path_img)
    print(all_image_paths)
    for file in all_image_paths:
        path_img_url = os.path.join(path_img, file)
        imgs.append(mpimg.imread(path_img_url))
        labels.append(file[0])
        n += 1
        if n > 20:
            break

    X = np.asarray(imgs, np.float32)
    y = np.asarray(labels, np.float32)
    print('=== dataset is ready, loading...')

    # split train and test
    train_images, test_images, train_labels, test_labels = train_test_split(X, y, test_size=0.4,
                                                                            random_state=42)
    # class name!!!
    label_names = ['no', 'yes']

    # Explore the data
    print('train_x: {},\ntrain_y: {},\ntest_x: {},\ntest_y:{},'.format(
        train_images.shape,
        len(train_labels),
        test_images.shape,
        len(test_labels))
    )

    # Preprocess the data
    train_images = train_images / 255
    test_images = test_images / 255

    # Build the model
    # - Set up the layers
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=train_images[0].shape),
        keras.layers.Dense(train_images[1].shape[2], activation='relu'),
        keras.layers.Dense(len(label_names))
    ])

    # - Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    # - Feed the model
    model.fit(train_images, train_labels, epochs=10)

    # 显示网络结构
    print(model.summary())

    # - Evaluate accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    # # - Make predictions
    # probability_model = tf.keras.Sequential([model,
    #                                          tf.keras.layers.Softmax()])
    # predictions = probability_model.predict(test_images)
    # print(predictions[0])
    #
    # # - Verify predictions
    #
    # # Use the trained model
    # # Grab an image from the test dataset.
    # img = test_images[1]
    # print('\ngrab an test_image: ', img.shape)
    # # Add the image to a batch where it's the only member.
    # img = (np.expand_dims(img, 0))
    # print(img.shape)
    # predictions_single = probability_model.predict(img)
    # print(predictions_single[0].max(), ', ', test_labels[1])

    # 将整个模型保存为HDF5文件，非常大的！
    # !pip install -q pyyaml h5py  # 需要以 HDF5 格式保存模型
    model.save(os.path.join(os.path.dirname(__file__), 'new_model_{}_{}.h5'.format(round(test_acc, 4), time.time())))

    # 重新创建完全相同的模型，包括其权重和优化程序
    # model = keras.models.load_model('')
    # print(model.summary())
    #

    return None


if __name__ == '__main__':
    my_url = 'img_tidy_by_class'
    # build_model_all(my_url)
    build_model_batch(my_url)
