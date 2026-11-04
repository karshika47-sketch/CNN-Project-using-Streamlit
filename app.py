import tensorflow as tf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# --- PART A: CNN IMAGE CLASSIFICATION (MNIST) ---

# 1. Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Preprocess the images
# Reshape to include the single color channel (grayscale)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

print("Data loaded and preprocessed!")
print(f"Training data shape: {X_train.shape}")
# 3. Create the CNN model
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # 10 classes for digits 0-9
])

# 4. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary (Take a screenshot of the terminal output for your PDF report!)
print("\n--- MODEL SUMMARY ---")
model.summary()

# 5. Train the model
print("\nTraining the model... (This might take a minute or two)")
# We use just 3 epochs to keep training fast for your assignment
history = model.fit(X_train, y_train, epochs=3, validation_split=0.2)

print("\nModel training complete!")
# 6. Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# 7. Plot Accuracy and Loss curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

# Loss plot
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

print("\nOpening training graphs. Please take a screenshot for your report, then close the window to continue...")
#plt.show() 

# 8. Show Sample Predictions (Actual vs Predicted)
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Plot a grid of 6 images
fig2, axes = plt.subplots(2, 3, figsize=(10, 6))
fig2.suptitle("Sample Predictions (Green=Correct, Red=Incorrect)", fontsize=14)
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    actual = y_test[i]
    predicted = predicted_classes[i]
    color = 'green' if actual == predicted else 'red'
    ax.set_title(f"Actual: {actual} | Pred: {predicted}", color=color)
    ax.axis('off')

plt.tight_layout()
print("\nOpening sample predictions. Please take a screenshot for your report, then close the window...")
#plt.show()
# --- PART B: STREAMLIT APPLICATION (Student Marks Calculator) ---

# Streamlit interface starts here
st.divider() # Adds a visual line to separate the CNN output from the app
st.title("🎓 Student Marks Calculator")
st.write("Enter your marks for three subjects to calculate your total, percentage, and final grade.")

# User Inputs with basic invalid-input handling (min_value=0, max_value=100)
subject1 = st.number_input("Mathematics (0-100)", min_value=0, max_value=100, step=1, value=0)
subject2 = st.number_input("Science (0-100)", min_value=0, max_value=100, step=1, value=0)
subject3 = st.number_input("English (0-100)", min_value=0, max_value=100, step=1, value=0)

# Useful processing and calculation triggered by a button
if st.button("Calculate Results"):
    total_marks = subject1 + subject2 + subject3
    max_marks = 300
    percentage = (total_marks / max_marks) * 100

    # Clean, understandable layout for outputs
    st.subheader("Your Results:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Marks", value=f"{total_marks} / {max_marks}")
    with col2:
        st.metric(label="Percentage", value=f"{percentage:.2f}%")

    # Final grade logic
    if percentage >= 50:
        st.success("Result: PASS 🎉")
    else:
        st.error("Result: FAIL")