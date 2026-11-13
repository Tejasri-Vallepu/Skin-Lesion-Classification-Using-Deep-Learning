
"""Check Python, TensorFlow, and available GPU devices."""
import platform
import sys

print("Python:", sys.version)
print("Platform:", platform.platform())

try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
except Exception as exc:
    print("TensorFlow check failed:", exc)
    print("Install dependencies using: pip install -r requirements.txt")
