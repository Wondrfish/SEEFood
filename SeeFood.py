import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Activation, LeakyReLU
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
import os

# Fix: Use double backslashes (\\) or raw strings (r"")
train_dir = r"C:\Users\ajani\OneDrive\Desktop\food_dataset\FOOD-11\training"
validation_dir = r"C:\Users\ajani\OneDrive\Desktop\food_dataset\FOOD-11\validation"
test_dir = r"C:\Users\ajani\OneDrive\Desktop\food_dataset\FOOD-11\evaluation"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,  
    brightness_range=[0.8, 1.2],  
    channel_shift_range=50.0,  
    fill_mode='nearest',  
    validation_split=0.2  # 80-20 split
)

# Load Training and Validation Data
train_generator = train_datagen.flow_from_directory(
    train_dir,  #
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    validation_dir,  
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Define CNN Model
model = Sequential([

    Conv2D(32, (3,3), padding='same', activation=None, input_shape=(224, 224, 3)),  
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), padding='same', activation=None),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), padding='same', activation=None),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(16, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(8, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation=None),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation='softmax') 
])

# Learning Rate Scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

# Compile Model with Adam Optimizer and Learning Rate Schedule
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Print Model Summary
model.summary()

# Train the Model
history = model.fit(
    train_generator,
    epochs=25,  # Increased Epochs
    validation_data=val_generator,
    callbacks=[early_stopping, lr_scheduler]
)

# Save the Model
model.save("food_classifier_model.h5")
print("Model saved successfully!")