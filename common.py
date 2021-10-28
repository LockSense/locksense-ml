from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_datasets(train_path, validation_path, target_size=(224, 224)):
    # Include data augmentation techniques
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        brightness_range=(0.5, 1.5),
		rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Resize to the required input shape
    batch_size = 8
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator


def model_callbacks(model_file, patience=10):
    checkpoint = ModelCheckpoint(model_file)
    earlystopping = EarlyStopping(min_delta=0.001, patience=patience)

    return [checkpoint, earlystopping]
