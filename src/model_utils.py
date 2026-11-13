
"""Model builders and training callbacks."""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.config import IMAGE_SIZE, CLASS_NAMES

def build_custom_cnn():
    """Build a CNN with BatchNormalization and dropout regularization."""
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = inputs

    # Three convolution blocks learn increasingly complex visual features.
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.20)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return models.Model(inputs, outputs, name="ham10000_custom_cnn")

def build_mobilenetv2_model():
    """Build a transfer-learning model using ImageNet-pretrained MobileNetV2."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    # MobileNetV2 preprocessing maps pixel values to its expected range.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = models.Model(inputs, outputs, name="ham10000_mobilenetv2")
    return model, base

def make_callbacks(model_path):
    """Create callbacks that reduce overfitting and save the best checkpoint."""
    return [
        EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            model_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1
        ),
    ]
