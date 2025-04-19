import sys
import time
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

def typewriter(text, color=Fore.WHITE, delay=0.05):
    for char in text:
        sys.stdout.write(color + char)
        sys.stdout.flush()
        time.sleep(delay)
    print(Style.RESET_ALL)

def animated_border(text, color=Fore.CYAN):
    border = "=" * (len(text) + 8)
    for c in border:
        sys.stdout.write(color + c)
        sys.stdout.flush()
        time.sleep(0.01)
    print()
    sys.stdout.write(color + "==  ")
    sys.stdout.flush()
    typewriter(text, color=color, delay=0.03)
    sys.stdout.write(color + "==")
    print()
    for c in border:
        sys.stdout.write(color + c)
        sys.stdout.flush()
        time.sleep(0.01)
    print(Style.RESET_ALL)

def decoration():
    animated_border("Welcome To AI Powered Anomaly Detection System", color=Fore.MAGENTA)
    time.sleep(0.5)
    typewriter("Initializing modules...", color=Fore.YELLOW)
    time.sleep(0.5)
    typewriter("System Ready! 🚀", color=Fore.GREEN)
    print()


def signup():
    print("\t\tPlease phly register krlo")
    email = input("Enter your email:")
    password = input("Enter your password:")
    name = input("Enter your name:")
    if "@gmail.com" in email:
        with open('credentials.txt', 'w') as file:
            file.write(f"{name},{email},{password}")
    else:
        print("Only Gmail addresses are accepted.")

def signin():
    print("\t\t\tSign In")
    global signin_control
    email = input("Enter your email:")
    password = input("Enter your password:")
    try:
        with open("credentials.txt", 'r') as file:
            credentials = file.readlines()
            for line in credentials:
                stored_name, stored_email, stored_password = line.strip().split(',')
                if email == stored_email and password == stored_password:
                    print("Sign in successful!")
                    signin_control = True
                    return
        print("Invalid email or password.")
    except FileNotFoundError:
        print("No credentials found. Please sign up first.")


decoration()
signup()
signin()

if(signin_control == False):
    exit


# Data processing and model code here (unchanged)
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load data
cv_data = pd.read_csv('J:\\4-Fourth Smester\\ADBMS Lab\\project\\data\\cv_server_data.csv', header=None, names=['feature1', 'feature2'])
tr_data = pd.read_csv('J:\\4-Fourth Smester\\ADBMS Lab\\project\\data\\tr_server_data.csv', header=None, names=['feature1', 'feature2'])
gt_data = pd.read_csv('J:\\4-Fourth Smester\\ADBMS Lab\\project\\data\\gt_server_data.csv', header=None, names=['label'])

# Check data shapes
print(f"CV data shape: {cv_data.shape}")
print(f"TR data shape: {tr_data.shape}")
print(f"GT data shape: {gt_data.shape}")

# Combine features
X = pd.concat([cv_data, tr_data], axis=0)

# Adjust labels
if len(gt_data) < len(X):
    repeat_times = len(X) // len(gt_data) + 1 # implementing division with floor operator. like 7//2 = 3
    y = pd.concat([gt_data] * repeat_times, axis=0).iloc[:len(X)]
else:
    y = gt_data.iloc[:len(X)]

print(f"Final X shape: {X.shape}")
print(f"Final y shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Calculate contamination
contamination = float(y_train.sum()) / len(y_train)
print(f"Estimated contamination: {contamination:.4f}")

model = IsolationForest(
    n_estimators=100,
    max_samples='auto',
    contamination=contamination,
    random_state=42,
    verbose=0
)
model.fit(X_train)

# Predict test set
y_pred = model.predict(X_test)
y_pred = np.where(y_pred == -1, 1, 0)

# Evaluate
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualization
def plot_anomalies(X, y, model):
    plt.figure(figsize=(10, 8))
    xx, yy = np.meshgrid(
        np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 100),
        np.linspace(X.iloc[:, 1].min(), X.iloc[:, 1].max(), 100)
    )
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 10), cmap=plt.cm.Blues_r, alpha=0.8)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
    
    normal_points = X[y.values == 0]
    anomalies_points = X[y.values == 1]
    
    plt.scatter(normal_points.iloc[:, 0], normal_points.iloc[:, 1], c='green', s=20, edgecolor='k', label='Normal')
    plt.scatter(anomalies_points.iloc[:, 0], anomalies_points.iloc[:, 1], c='red', s=30, edgecolor='k', label='Anomaly')
    
    plt.title("Anomaly Detection Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

plot_anomalies(X_test, y_test, model)

# Function to detect anomalies on new data
def detect_anomalies(new_data, model):
    scores = model.decision_function(new_data)
    preds = model.predict(new_data)
    anomalies = np.where(preds == -1, 1, 0)
    result = new_data.copy()
    result['anomaly'] = anomalies
    result['anomaly_score'] = scores
    return result

# Run user authentication and detection

test_results = detect_anomalies(X_test, model)
print("\nFirst 5 test results:")
print(test_results.head())
print("Please register first.")