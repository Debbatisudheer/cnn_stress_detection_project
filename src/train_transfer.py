# train_transfer.py

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------- 1. Data Loader with Augmentation ----------
def load_data(train_dir, test_dir, img_size=(224,224), batch_size=32):
    # Image augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        color_mode='rgb',       # RGB for pre-trained models
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        color_mode='rgb',       # RGB for pre-trained models
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, test_generator

# ---------- 2. Model Definition (VGG16 Transfer Learning) ----------
def build_transfer_model(num_classes=7):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    
    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom top layers
    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------- 3. Training Function ----------
def train_model(train_dir, test_dir, saved_model_path="saved_models/transfer_model.h5", epochs=20):
    train_gen, test_gen = load_data(train_dir, test_dir, img_size=(224,224))

    model = build_transfer_model(num_classes=len(train_gen.class_indices))

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ModelCheckpoint(saved_model_path, monitor='val_accuracy', save_best_only=True)
    ]

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=test_gen,
        callbacks=callbacks
    )

    print(f"Training Completed. Model saved in {saved_model_path}")
    return model, history

# ---------- 4. Run Training ----------
if __name__ == "__main__":
    train_dir = "../archive/train"  # path to your train folder
    test_dir = "../archive/test"    # path to your test folder
    os.makedirs("saved_models", exist_ok=True)

    train_model(train_dir, test_dir, epochs=20)
