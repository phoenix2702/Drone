import tensorflow as tf
model = tf.keras.models.load_model("obstacle_detector.h5")
model.save("compressed_model.h5", include_optimizer=False)
