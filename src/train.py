from src.data_loader import load_data
from .model import build_cnn
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import os

def train_model(train_dir, test_dir, saved_model_path="saved_models/cnn_stress_model.h5", resume=False):
    train_gen, test_gen = load_data(train_dir, test_dir)

    # ✅ If resume=True and model exists, load it, otherwise build a new one
    if resume and os.path.exists(saved_model_path):
        print(f"Resuming training from saved model: {saved_model_path}")
        model = load_model(saved_model_path)
    else:
        print("Starting training from scratch...")
        model = build_cnn()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        ModelCheckpoint(saved_model_path, monitor='val_accuracy', save_best_only=True)
    ]

    history = model.fit(
        train_gen,
        epochs=50,   # you can change this to 100, 200, etc.
        validation_data=test_gen,
        callbacks=callbacks
    )

    return model, history
