from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Формирование обучающей и тестирующей выборок
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels_flat, test_size=0.2, random_state=42)

# Список моделей для обучения
models = [
    ("SVM", SVC(max_iter=1000)),
    ("KNN", KNeighborsClassifier()),
    ("Naive Bayes", GaussianNB()),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("Random Forest", RandomForestClassifier(n_estimators=50, random_state=42)),
    ("Bagging", BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)),
    ("MLP", MLPClassifier(max_iter=300))
]

# Обучение и оценка моделей
model_accuracies = []

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    model_accuracies.append((name, accuracy, precision, recall, f1))
    print(f"{name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

# Визуализация результатов
fig, ax = plt.subplots()
for name, accuracy, precision, recall, f1 in model_accuracies:
    ax.plot(range(1, 8), [accuracy, precision, recall, f1], label=name)

ax.set_xlabel("Метрики")
ax.set_ylabel("Значения")
ax.set_title("Точность предсказаний моделей на тестовом наборе данных")
ax.legend()
plt.savefig('model_performance.png')
plt.show()
