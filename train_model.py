import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Read the dataset
df = pd.read_csv('iris.csv')

# Prepare the input features and target variable
X = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = df['variety']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, pd.get_dummies(y_train), epochs=200, batch_size=10)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, pd.get_dummies(y_test))
print(f'Test accuracy: {accuracy}')

# Save the trained model
model.save('iris_model1.h5')