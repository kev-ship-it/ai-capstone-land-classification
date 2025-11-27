import os
from typing import Tuple
import urllib.request

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="Custom")
class AddPositionEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.pos = self.add_weight(
            name="pos_embedding",
            shape=(1, num_patches, embed_dim),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, tokens):
        return tokens + self.pos

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_patches": self.num_patches,
                "embed_dim": self.embed_dim,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, mlp_dim=2048, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential(
            [
                layers.Dense(mlp_dim, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(embed_dim),
                layers.Dropout(dropout),
            ]
        )

    def call(self, x):
        x = x + self.mha(self.norm1(x), self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
            }
        )
        return config


# Local path where model will be stored in the container
MODEL_PATH = os.environ.get("KERAS_VIT_PATH", "./keras_cnn_vit_ai_capstone.keras")

# Google Drive direct-download URL (your file ID)
MODEL_URL = os.environ.get(
    "KERAS_VIT_URL",
    "https://drive.google.com/uc?export=download&id=1ZHTFMXuZWNFMId2QFeA9ZVpoz0221P3n",
)

IMG_SIZE = 64
_model = None


def _ensure_model_file():
    """Download the model from Google Drive if it's not present locally."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def load_keras_vit():
    global _model
    if _model is None:
        _ensure_model_file()
        _model = load_model(
            MODEL_PATH,
            custom_objects={
                "AddPositionEmbedding": AddPositionEmbedding,
                "TransformerBlock": TransformerBlock,
            },
        )
    return _model


def preprocess_keras_image(file_obj) -> np.ndarray:
    img = Image.open(file_obj).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_keras_vit(file_obj) -> Tuple[float, str]:
    model = load_keras_vit()
    x = preprocess_keras_image(file_obj)
    preds = model.predict(x, verbose=0)[0]
    p_agri = float(preds[0]) if getattr(preds, "shape", None) else float(preds)
    label = "agri" if p_agri >= 0.5 else "non-agri"
    return p_agri, label
