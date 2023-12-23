from sklearn.neural_network import MLPClassifier
from utils import load_data

# Load your training data
X_train, X_test, y_train, y_test = load_data(test_size=0.2)

# Create and train the MLPClassifier model
model = MLPClassifier()
model.fit(X_train, y_train)

# Save the trained model
import pickle
with open("mlp_classifier_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
