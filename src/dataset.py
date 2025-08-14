from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(train_dir="train/", test_dir="test/", img_size=(48, 48), batch_size=32):
    # Training data with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        validation_split=0.2  # 20% for validation
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
