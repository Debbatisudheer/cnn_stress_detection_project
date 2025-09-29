from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# class names in the same order as training
class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def predict_emotion(img_path, model_path="saved_models/cnn_stress_model.h5", img_size=(48,48)):
    # load trained model
    model = load_model(model_path)

    # load image as grayscale
    img = image.load_img(img_path, target_size=img_size, color_mode="grayscale")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape -> (1,48,48,1)

    # predict
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)

    return class_names[class_idx], predictions[0]

if __name__ == "__main__":
    label, probs = predict_emotion("sample_images/cry1.webp")
    print("Predicted emotion:", label)
    print("Probabilities:", probs)
