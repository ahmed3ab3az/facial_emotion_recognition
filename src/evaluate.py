import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.dataset import get_data_generators

def evaluate_model(model_path="models/best_vgg16_model.h5", batch_size=32):
    # Load the test data
    _, _, test_generator = get_data_generators(batch_size=batch_size)

    # Load  model
    model = tf.keras.models.load_model(model_path)

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    test_generator.reset()
    predictions = model.predict(test_generator, steps=test_generator.samples // batch_size + 1)
    predicted_classes = np.argmax(predictions, axis=1)

    # True labels
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_classes, predicted_classes))

if __name__ == "__main__":
    evaluate_model()
