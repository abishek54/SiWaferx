import tensorflow as tf
import tf2onnx

# Load trained model
model = tf.keras.models.load_model("sem_defect_cnn_final.keras")

# Define input signature
spec = (tf.TensorSpec((None, 128, 128, 1), tf.float32, name="input"),)

# Convert to ONNX
output_path = "sem_defect_cnn_final.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

print("✅ Model successfully exported to ONNX:", output_path)
