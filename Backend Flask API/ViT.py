    

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class ClassToken(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value = w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable = True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls

def mlp(x, cf):
    x = Dense(cf["mlp_dim"], activation="gelu")(x)
    x = Dropout(cf["dropout_rate"])(x)
    x = Dense(cf["hidden_dim"])(x)
    x = Dropout(cf["dropout_rate"])(x)
    return x

def transformer_encoder(x, cf):
    skip_1 = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(
        num_heads=cf["num_heads"], key_dim=cf["hidden_dim"]
    )(x, x)
    x = Add()([x, skip_1])

    skip_2 = x
    x = LayerNormalization()(x)
    x = mlp(x, cf)
    x = Add()([x, skip_2])

    return x

def ViT(cf):
    """ Inputs """
    input_shape = (cf["num_patches"], cf["patch_size"]*cf["patch_size"]*cf["num_channels"])
    inputs = Input(input_shape)     ## (None, 256, 3072)

    """ Patch + Position Embeddings """
    patch_embed = Dense(cf["hidden_dim"])(inputs)   ## (None, 256, 768)

    positions = tf.range(start=0, limit=cf["num_patches"], delta=1)
    pos_embed = Embedding(input_dim=cf["num_patches"], output_dim=cf["hidden_dim"])(positions) ## (256, 768)
    embed = patch_embed + pos_embed ## (None, 256, 768)

    """ Adding Class Token """
    token = ClassToken()(embed)
    x = Concatenate(axis=1)([token, embed]) ## (None, 257, 768)

    for _ in range(cf["num_layers"]):
        x = transformer_encoder(x, cf)

    """ Classification Head """
    x = LayerNormalization()(x)     ## (None, 257, 768)
    x = x[:, 0, :]
    x = Dense(cf["num_classes"], activation="softmax")(x)

    model = Model(inputs, x)
    return model


import numpy as np
import cv2
from patchify import patchify
import tensorflow as tf
import random


class Pred():
    def __init__(self,model_path):
        """ Hyperparameters """
        self.hp = {}
        self.hp["w"]=700
        self.hp["l"]=256
        self.hp["num_channels"] = 3
        self.hp["patch_size"] = 16
        self.hp["num_patches_w"] = self.hp["w"] // self.hp["patch_size"]
        self.hp["num_patches_l"] = self.hp["l"] // self.hp["patch_size"]
        self.hp["num_patches"] = self.hp["num_patches_w"] * self.hp["num_patches_l"]
        self.hp["flat_patches_shape"] = (self.hp["num_patches"], self.hp["patch_size"]*self.hp["patch_size"]*self.hp["num_channels"])

        self.hp["batch_size"] = 8
        self.hp["lr"] = 1e-4
        self.hp["num_epochs"] = 10
        self.hp["num_classes"] = 2
        self.hp["class_names"] = ["healthy","unhealthy"]

        self.hp["num_layers"] = 12
        self.hp["hidden_dim"] = 512
        self.hp["mlp_dim"] = 3072
        self.hp["num_heads"] = 12
        self.hp["dropout_rate"] = 0.1
        

        self.model = ViT(self.hp)
        self.model.summary()
        self.model.load_weights(model_path)
        print("Model object loaded")

    def process_image(self, path):
        """ Reading images """
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.hp["l"], self.hp["w"]))  # Notice the swap in dimensions
        image = image / 255.0

        """ Preprocessing to patches """
        patch_shape = (self.hp["patch_size"], self.hp["patch_size"], self.hp["num_channels"])
        patches = patchify(image, patch_shape, self.hp["patch_size"])

        patches = patches.reshape((self.hp["num_patches"], self.hp["patch_size"]*self.hp["patch_size"]*self.hp["num_channels"]))
        patches = patches.astype(np.float32)

        return patches
    
    def predict(self, image_path):
        patches = self.process_image(image_path)

    #     """ Make Prediction """
        patches = np.expand_dims(patches, axis=0)  # Add batch dimension
        prediction = self.model.predict(patches)

        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        predicted_class = self.hp["class_names"][predicted_class_idx]
        # print(f"Predicted class: {predicted_class}")
        # print(f"Prediction: {prediction[0]}")
        return predicted_class, prediction[0]


