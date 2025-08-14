import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(
    base_data_dir="../data",  
    train_folder="train",
    test_folder="test",
    img_size=(48, 48),
    batch_size=32
):
    train_dir = os.path.join(base_data_dir, train_folder)
    test_dir = os.path.join(base_data_dir, test_folder)

    # Training data with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        validation_split=0.2
    )

    # Test data without augmentation
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Train generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Test generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator
