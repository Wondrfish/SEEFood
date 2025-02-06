import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import MobileNetV2
import os

# Fix: Use double backslashes (\\) or raw strings (r"")
train_dir = r"training"
validation_dir = r"validation"
test_dir = r"evaluation"

# Data Augmentation
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

# Load Training Data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# **Fix: Create Validation Data**
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# **Step 1: Load Pre-trained MobileNetV2 as the Base Model**
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# **Step 2: Freeze the Base Model for Initial Training**
base_model.trainable = False

# **Step 3: Build the Model**
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation=None),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation='softmax') 
])

# Learning Rate Scheduler & Early Stopping
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# Step 4: Compile and Train the Model (Initial Training)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=20,  
    validation_data=val_generator,  # ✅ Fixed missing validation generator
    callbacks=[lr_scheduler, early_stopping]
)

# Save the Initial Model
model.save("food_classifier_mobilenetv2_initial.h5")
print("Initial model saved successfully!")

# Step 5: Fine-Tune by Unfreezing Some Layers
for layer in base_model.layers[-20:]:  # Unfreezing last 20 layers
    layer.trainable = True

# Step 6: Recompile with a Lower Learning Rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Step 7: Fine-Tune the Model
history_finetune = model.fit(
    train_generator,
    epochs=20,  
    validation_data=val_generator,  # ✅ Fixed missing validation generator
    callbacks=[lr_scheduler, early_stopping]
)

# Step 8: Load Test Data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Step 9: Evaluate the Fine-Tuned Model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Final Test Accuracy: {test_acc:.4f}")

# Step 10: Save the Fine-Tuned Model
model.save("food_classifier_mobilenetv2_finetuned.h5")
print("Fine-tuned model saved successfully!")
