from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model_path, test_generator):
    model = load_model(model_path)
    test_generator.reset()

    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
