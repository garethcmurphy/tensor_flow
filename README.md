# TensorFlow Playground with Iris Dataset 🌸🤖  

This repository provides a playground for experimenting with **TensorFlow** using the classic **Iris dataset**. It is designed to help users get familiar with TensorFlow concepts and workflows in Python, including data preprocessing, model training, and evaluation.

---

## Features ✨  

- **Introduction to TensorFlow**: Practice fundamental TensorFlow workflows.  
- **Iris Dataset Example**: Explore classification with a well-known dataset.  
- **Customizable Models**: Modify and experiment with different architectures and hyperparameters.  

---

## Prerequisites 🛠️  

- Python 3.8+  
- Required Python libraries:
  - `tensorflow`
  - `numpy`
  - `pandas`
  - `matplotlib`  

Install dependencies:  
pip install tensorflow numpy pandas matplotlib  

---

## Installation  

1. Clone the repository:  
git clone https://github.com/your-username/tensorflow-playground.git  
cd tensorflow-playground  

2. Install required dependencies:  
pip install -r requirements.txt  

---

## Usage 🔧  

1. Open the Python script:  
`iris_tensorflow.py`  

2. Run the script to train a model:  
python iris_tensorflow.py  

3. View model performance metrics and visualizations generated during training.  

---

## File Structure 📂  

- `iris_tensorflow.py`: Main Python script for training and evaluating the TensorFlow model.  
- `README.md`: Documentation for the repository.  

---

## Example Code Snippet  

```python
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load and preprocess data
data = load_iris()
X = data.data
y = OneHotEncoder().fit_transform(data.target.reshape(-1, 1)).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")
