import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Завантаження даних з CSV файлу
data = pd.read_csv("bioresponse.csv")

# Розділімо дані на ознаки (X) і цільову змінну (y)
X = data.drop("Activity", axis=1)
y = data["Activity"]

# Розділімо дані на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Навчання дрібного дерева рішень
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Навчання глибокого дерева рішень
deep_dt_classifier = DecisionTreeClassifier(max_depth=10, random_state=42)
deep_dt_classifier.fit(X_train, y_train)

# Навчання випадкового лісу на дрібних деревах
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_classifier.fit(X_train, y_train)

# Навчання випадкового лісу на глибоких деревах
deep_rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
deep_rf_classifier.fit(X_train, y_train)

# Передбачення класів на тестовому наборі для кожної моделі
y_pred_dt = dt_classifier.predict(X_test)
y_pred_deep_dt = deep_dt_classifier.predict(X_test)
y_pred_rf = rf_classifier.predict(X_test)
y_pred_deep_rf = deep_rf_classifier.predict(X_test)

# Оцінка якості моделей
accuracy_dt = accuracy_score(y_test, y_pred_dt)
accuracy_deep_dt = accuracy_score(y_test, y_pred_deep_dt)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_deep_rf = accuracy_score(y_test, y_pred_deep_rf)

precision_dt = precision_score(y_test, y_pred_dt)
precision_deep_dt = precision_score(y_test, y_pred_deep_dt)
precision_rf = precision_score(y_test, y_pred_rf)
precision_deep_rf = precision_score(y_test, y_pred_deep_rf)

recall_dt = recall_score(y_test, y_pred_dt)
recall_deep_dt = recall_score(y_test, y_pred_deep_dt)
recall_rf = recall_score(y_test, y_pred_rf)
recall_deep_rf = recall_score(y_test, y_pred_deep_rf)

f1_score_dt = f1_score(y_test, y_pred_dt)
f1_score_deep_dt = f1_score(y_test, y_pred_deep_dt)
f1_score_rf = f1_score(y_test, y_pred_rf)
f1_score_deep_rf = f1_score(y_test, y_pred_deep_rf)

log_loss_dt = log_loss(y_test, dt_classifier.predict_proba(X_test)[:, 1])
log_loss_deep_dt = log_loss(y_test, deep_dt_classifier.predict_proba(X_test)[:, 1])
log_loss_rf = log_loss(y_test, rf_classifier.predict_proba(X_test)[:, 1])
log_loss_deep_rf = log_loss(y_test, deep_rf_classifier.predict_proba(X_test)[:, 1])

# Виведення результатів
print("Дрібне дерево рішень:")
print(f"Accuracy: {accuracy_dt}")
print(f"Precision: {precision_dt}")
print(f"Recall: {recall_dt}")
print(f"F1 Score: {f1_score_dt}")
print(f"Log Loss: {log_loss_dt}")
print("\n")

print("Глибоке дерево рішень:")
print(f"Accuracy: {accuracy_deep_dt}")
print(f"Precision: {precision_deep_dt}")
print(f"Recall: {recall_deep_dt}")
print(f"F1 Score: {f1_score_deep_dt}")
print(f"Log Loss: {log_loss_deep_dt}")
print("\n")

print("Випадковий ліс на дрібних деревах:")
print(f"Accuracy: {accuracy_rf}")
print(f"Precision: {precision_rf}")
print(f"Recall: {recall_rf}")
print(f"F1 Score: {f1_score_rf}")
print(f"Log Loss: {log_loss_rf}")
print("\n")

print("Випадковий ліс на глибоких деревах:")
print(f"Accuracy: {accuracy_deep_rf}")
print(f"Precision: {precision_deep_rf}")
print(f"Recall: {recall_deep_rf}")
print(f"F1 Score: {f1_score_deep_rf}")
print(f"Log Loss: {log_loss_deep_rf}")
print("\n")


# Побудова precision-recall і ROC-кривих для кожної моделі
def plot_precision_recall_curve(y_true, y_prob, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve для {model_name}')
    plt.show()


def plot_roc_curve(y_true, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.show()


# Для дрібного дерева рішень
y_prob_dt = dt_classifier.predict_proba(X_test)[:, 1]
plot_precision_recall_curve(y_test, y_prob_dt, "Дрібне дерево рішень")
plot_roc_curve(y_test, y_prob_dt, "Дрібне дерево рішень")

# Для глибокого дерева рішень
y_prob_deep_dt = deep_dt_classifier.predict_proba(X_test)[:, 1]
plot_precision_recall_curve(y_test, y_prob_deep_dt, "Глибоке дерево рішень")
plot_roc_curve(y_test, y_prob_deep_dt, "Глибоке дерево рішень")

# Для випадкового лісу на дрібних деревах
y_prob_rf = rf_classifier.predict_proba(X_test)[:, 1]
plot_precision_recall_curve(y_test, y_prob_rf, "Випадковий ліс на дрібних деревах")
plot_roc_curve(y_test, y_prob_rf, "Випадковий ліс на дрібних деревах")

# Для випадкового лісу на глибоких деревах
y_prob_deep_rf = deep_rf_classifier.predict_proba(X_test)[:, 1]
plot_precision_recall_curve(y_test, y_prob_deep_rf, "Випадковий ліс на глибоких деревах")
plot_roc_curve(y_test, y_prob_deep_rf, "Випадковий ліс на глибоких деревах")

# Навчання класифікатора, який уникає помилок II роду
# Використаємо метод SMOTE для балансування класів
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Навчання класифікатора (можете використовувати той самий, як і в попередніх моделях)
# Передбачення на тестовому наборі
y_pred_balanced_dt = dt_classifier.predict(X_test)
y_pred_balanced_deep_dt = deep_dt_classifier.predict(X_test)
y_pred_balanced_rf = rf_classifier.predict(X_test)
y_pred_balanced_deep_rf = deep_rf_classifier.predict(X_test)

# Оцінка якості класифікатора, що уникає помилок II роду
confusion_matrix_balanced_dt = confusion_matrix(y_test, y_pred_balanced_dt)
classification_report_balanced_dt = classification_report(y_test, y_pred_balanced_dt)

confusion_matrix_balanced_deep_dt = confusion_matrix(y_test, y_pred_balanced_deep_dt)
classification_report_balanced_deep_dt = classification_report(y_test, y_pred_balanced_deep_dt)

confusion_matrix_balanced_rf = confusion_matrix(y_test, y_pred_balanced_rf)
classification_report_balanced_rf = classification_report(y_test, y_pred_balanced_rf)

confusion_matrix_balanced_deep_rf = confusion_matrix(y_test, y_pred_balanced_deep_rf)
classification_report_balanced_deep_rf = classification_report(y_test, y_pred_balanced_deep_rf)

print("Confusion Matrix for Balanced dt Classifier:")
print(confusion_matrix_balanced_dt)
print("\nClassification Report for Balanced dt Classifier:")
print(classification_report_balanced_dt)

print("Confusion Matrix for Balanced deep dt Classifier:")
print(confusion_matrix_balanced_deep_dt)
print("\nClassification Report for Balanced deep dt Classifier:")
print(classification_report_balanced_deep_dt)

print("Confusion Matrix for Balanced rf Classifier:")
print(confusion_matrix_balanced_rf)
print("\nClassification Report for Balanced rf Classifier:")
print(classification_report_balanced_rf)

print("Confusion Matrix for Balanced deep rf Classifier:")
print(confusion_matrix_balanced_deep_rf)
print("\nClassification Report for Balanced deep rf Classifier:")
print(classification_report_balanced_deep_rf)
