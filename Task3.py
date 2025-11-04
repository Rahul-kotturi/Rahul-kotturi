import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog

# Path to dataset (make sure this is correct)
DATA_DIR = r"C:\Users\Rahul\Desktop\Intern\Dataset"

IMG_SIZE = (128, 128)

def load_images_and_labels(data_dir):
    images = []
    labels = []

    for label, cls in enumerate(["cats", "dogs"]):
        folder = os.path.join(data_dir, cls)
        for img_name in tqdm(os.listdir(folder), desc=f"Loading {cls}"):
            img_path = os.path.join(folder, img_name)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Resize
            img = cv2.resize(img, IMG_SIZE)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Extract HOG features
            hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), block_norm='L2-Hys')
            
            images.append(hog_features)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Load dataset
X, y = load_images_and_labels(DATA_DIR)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
model = LinearSVC()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))
