import tensorflow as tf
import sys

gpus = tf.config.list_physical_devices('GPU')

if not gpus:
    print("Critical Warning: No GPU was detected. TensorFlow will use the CPU.")
    sys.exit(1)
else:
    print("Success: GPU detected.")

for gpu in gpus:
    details = tf.config.experimental.get_device_details(gpu)
    device_name = details.get('device_name', 'Unknown GPU')
    print(device_name)

sys.exit(0)