import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Fix: Use double backslashes (\\) or raw strings (r"")
train_dir = r"training"
validation_dir = r"validation"
test_dir = r"evaluation"

# Fix: Use `train_dir` and `validation_dir` properly
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80-20 split
)

# Load Training and Validation Data
train_generator = train_datagen.flow_from_directory(
    train_dir,  # FIXED: Use correct directory
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    validation_dir,  # FIXED: Use correct directory
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(2,2), 

    GlobalAveragePooling2D(), 
    Dense(128, activation='relu'),
    Dropout(0.5), 
    Dense(train_generator.num_classes, activation='softmax') 
])

#  Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#  Print Model Summary
model.summary()

# Train the Model
model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator
)

# Save the Model
model.save("food_classifier_model.h5")
print("Model saved successfully!")