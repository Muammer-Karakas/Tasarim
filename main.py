import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt

# Kalori veri setini yükleme
calorie_file = "KaloriDs.txt"

def load_calorie_data(file_path):
    calorie_data = pd.read_csv(file_path, header=None, names=['food', 'calories'])
    return calorie_data

calorie_data = load_calorie_data(calorie_file)

# Görüntü boyutu
img_size = (128, 128)

# Görsellerden özellik çıkarma
def extract_features(image_path):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(img_size)
    features = np.array(img_resized).flatten()
    return features

# Veriyi hazırlama
def prepare_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, file)
                features = extract_features(img_path)
                images.append(features)
                labels.append(label)
    return np.array(images), np.array(labels)

# Eğitim ve test veri seti dizinleri
train_dir = "YemekEgitimDs"
test_dir = "YemekTest"

X_train, y_train = prepare_data(train_dir)
X_test, y_test = prepare_data(test_dir)

# Epoch ayarları
num_epochs = 15  # Epoch sayısı
batch_size = 64
acceptable_error = 0.01  # Kabul edilebilir hata oranı

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
calorie_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Eğitim döngüsü
best_test_accuracy = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Veriyi karıştırma
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # Modeli eğitme
    model.fit(X_train, y_train)

    # Eğitim doğruluğunu hesaplama (isteğe bağlı)
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"  Egitim Dogrulugu: {train_accuracy * 100:.2f}%")

    # Test doğruluğunu hesaplama
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"  Test Dogrulugu: {test_accuracy * 100:.2f}%")

    # Hata oranını kontrol etme
    test_error = 1 - test_accuracy
    if test_error <= acceptable_error:
        print(f"Hata orani kabul edilebilir seviyeye ulasti ({test_error:.4f}). Egitim durduruluyor.")
        break

    # Overfitting kontrolü: Test doğruluğu artmıyorsa eğitimi durdurma
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
    else:
        print("Test dogrulugu iyilesmedi. Egitim durduruluyor.")
        break

# Kalori tahmini ve hata hesaplama
def predict_calories(food_label):
    calories = calorie_data.loc[calorie_data['food'] == food_label, 'calories']
    return calories.values[0] if not calories.empty else None

# Regresyon modelini eğitme
calorie_labels = calorie_data['food'].map({food: idx for idx, food in enumerate(calorie_data['food'].unique())})
calorie_values = calorie_data['calories']
calorie_regressor.fit(calorie_labels.values.reshape(-1, 1), calorie_values)

y_true_calories = []
y_pred_calories = []

for true_label, predicted_label in zip(y_test, y_test_pred):
    true_calories = predict_calories(true_label)
    pred_calories = predict_calories(predicted_label)
    if true_calories is not None and pred_calories is not None:
        y_true_calories.append(true_calories)
        y_pred_calories.append(pred_calories)

calorie_mse = mean_squared_error(y_true_calories, y_pred_calories)
print(f"Kalori Tahmininde Ortalama Hata (MSE): {calorie_mse:.2f}")

# Test sonuçlarının kaydedilmesi
results = pd.DataFrame({
    'Gercek': y_test,
    'Tahmin': y_test_pred,
    'Gercek Kalori': y_true_calories,
    'Tahmin Kalori': y_pred_calories
})
results.to_csv("test_sonuclari.csv", index=False)
print("Test sonuclari 'test_sonuclari.csv' dosyasina kaydedildi.")

# Model performansı analizi
print("Classification Report:")
print(classification_report(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gercek')
plt.title('Confusion Matrix')
plt.show()

