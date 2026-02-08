import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


path = '/home/user/PycharmProjects/Samsung/project/data/processed/gestures.csv'
df = pd.read_csv(path)

X = df.iloc[:, 1:].values.astype('float32')
y = df.iloc[:, 0].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(63,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("AI-ի մարզումը սկսվեց...")
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test))

model.save('hand_model.h5')
print("\nՄոդելը հաջողությամբ պահպանվեց որպես 'hand_model.h5'")