import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import numpy as np

# Создание примеров данных
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, 100)

# Обучение Random Forest
rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
rf_model.fit(X_train, y_train)

# Обучение Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Обучение Stacking
estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, random_state=42))
]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_model.fit(X_train, y_train)

# Визуализация структуры Random Forest
def plot_random_forest_structure(filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Структура Random Forest')
    ax.text(0.5, 0.5, 'Random Forest\nс множеством деревьев решений', 
            horizontalalignment='center', verticalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    ax.axis('off')
    plt.savefig(filename)
    plt.close(fig)

# Визуализация структуры Gradient Boosting
def plot_gradient_boosting_structure(filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Структура Gradient Boosting')
    ax.text(0.5, 0.5, 'Gradient Boosting\nс последовательными деревьями решений', 
            horizontalalignment='center', verticalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    ax.axis('off')
    plt.savefig(filename)
    plt.close(fig)

# Визуализация структуры Stacking
def plot_stacking_structure(filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Структура Stacking')
    layers = [
        "Random Forest (базовый алгоритм)",
        "Gradient Boosting (базовый алгоритм)",
        "Logistic Regression (финальный алгоритм)"
    ]
    for i, layer in enumerate(layers):
        ax.text(0.5, 1 - (i * 0.3), layer, 
                horizontalalignment='center', verticalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    ax.axis('off')
    plt.savefig(filename)
    plt.close(fig)

# Визуализация структуры Stacking с деревом решений
def plot_stacking_with_tree(filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Структура Stacking')
    layers = [
        "Random Forest (базовый алгоритм)",
        "Gradient Boosting (базовый алгоритм)",
        "Logistic Regression (финальный алгоритм)",
        "Пример дерева решений из Random Forest"
    ]
    for i, layer in enumerate(layers):
        ax.text(0.5, 1 - (i * 0.2), layer, 
                horizontalalignment='center', verticalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    ax.axis('off')
    plt.savefig(filename)
    plt.close(fig)

# Визуализация структур моделей
plot_random_forest_structure('random_forest_structure.png')
plot_gradient_boosting_structure('gradient_boosting_structure.png')
plot_stacking_structure('stacking_structure.png')
# Визуализация структуры Stacking с деревом решений
plot_stacking_with_tree('stacking_with_tree.png')

plt.show()
