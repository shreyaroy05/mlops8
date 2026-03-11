from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    data = load_iris()
    x = data.data
    y = data.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    print("Model Accuracy:", accuracy)

    joblib.dump(model, "model.pkl")
    print("Model saved as model.pkl")


if __name__ == "__main__":
    print("Training started...")
    train_model()