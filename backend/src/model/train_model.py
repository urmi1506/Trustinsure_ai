import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from tensorflow.keras.callbacks import EarlyStopping

from src.preprocessing.preprocess import preprocess

# Load preprocessed data
X_train, X_test, y_train, y_test,preprocessor = preprocess()

# Build model
model = Sequential([
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    # Dense(32, activation='relu'),
    Dense(1)
])


# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)


# Make predictions
y_pred = model.predict(X_test)


# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

print("Initial training loss:", history.history["loss"][0])
print("Final training loss:", history.history["loss"][-1])

print("Initial validation loss:", history.history["val_loss"][0])
print("Final validation loss:", history.history["val_loss"][-1])

best_epoch = np.argmin(history.history["val_loss"]) + 1
print("Best epoch:", best_epoch)

model.save("src/model/insurance_model.keras")