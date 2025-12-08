import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay
import warnings

warnings.filterwarnings('ignore')


class AbaloneTreeModel:
    def __init__(self, data_path='abalone.data'):
        self.data_path = data_path
        self.column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
                             'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
        self.data = None
        self.X = None
        self.y = None
        self.X_scaled = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.tree_model = None
        self.best_params = None

    def load_and_preprocess(self):
        self.data = pd.read_csv(self.data_path, names=self.column_names)
        print(f"Данные загружены. Размер: {self.data.shape}")

        self.data['Sex'] = self.label_encoder.fit_transform(self.data['Sex'])
        self.X = self.data.drop('Rings', axis=1)

        bins = [0, 5, 10, 15, 20, 30]
        labels = ['очень молодой', 'молодой', 'средний', 'взрослый', 'старый']
        self.y = pd.cut(self.data['Rings'], bins=bins, labels=labels)

        self.X_scaled = self.scaler.fit_transform(self.X)

        print(f"Классы: {labels}")
        print(f"Распределение классов:\n{self.y.value_counts()}")

        return self.X_scaled, self.y

    def build_tree(self, max_depth=None, max_features=None):
        self.tree_model = DecisionTreeClassifier(
            max_depth=max_depth,
            max_features=max_features,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.3, random_state=42
        )

        self.tree_model.fit(X_train, y_train)

        train_acc = self.tree_model.score(X_train, y_train)
        test_acc = self.tree_model.score(X_test, y_test)

        print(f"Параметры дерева: max_depth={max_depth}, max_features={max_features}")
        print(f"Точность на обучающей выборке: {train_acc:.4f}")
        print(f"Точность на тестовой выборке: {test_acc:.4f}")

        y_pred = self.tree_model.predict(X_test)

        print("\nОтчет по классификации:")
        print(classification_report(y_test, y_pred))

        return self.tree_model

    def visualize_tree(self, filename='decision_tree.png'):
        if self.tree_model is None:
            print("Сначала постройте дерево")
            return

        plt.figure(figsize=(20, 10))
        plot_tree(
            self.tree_model,
            feature_names=self.column_names[:-1],
            class_names=self.tree_model.classes_.astype(str),
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title(f"Дерево решений (max_depth={self.tree_model.get_depth()})")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Дерево сохранено в {filename}")

    def cv_max_depth(self, depths=range(1, 21), cv_folds=5):
        accuracy_scores = []
        f1_scores = []

        for depth in depths:
            tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
            acc_scores = cross_val_score(tree, self.X_scaled, self.y,
                                         cv=cv_folds, scoring='accuracy')
            accuracy_scores.append(acc_scores.mean())

            f1_scores_cv = cross_val_score(tree, self.X_scaled, self.y,
                                           cv=cv_folds, scoring='f1_weighted')
            f1_scores.append(f1_scores_cv.mean())

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        axes[0].plot(depths, accuracy_scores, marker='o', linewidth=2, markersize=8, color='blue')
        axes[0].set_xlabel('max_depth')
        axes[0].set_ylabel('Accuracy (Cross-Validation)')
        axes[0].set_title('Зависимость Accuracy от max_depth')
        axes[0].grid(True, alpha=0.3)

        best_depth_acc = depths[np.argmax(accuracy_scores)]
        axes[0].axvline(x=best_depth_acc, color='r', linestyle='--',
                        label=f'Лучший max_depth = {best_depth_acc}')
        axes[0].legend()

        axes[1].plot(depths, f1_scores, marker='s', linewidth=2, markersize=8, color='green')
        axes[1].set_xlabel('max_depth')
        axes[1].set_ylabel('F1 Score (Weighted)')
        axes[1].set_title('Зависимость F1 Score от max_depth')
        axes[1].grid(True, alpha=0.3)

        best_depth_f1 = depths[np.argmax(f1_scores)]
        axes[1].axvline(x=best_depth_f1, color='r', linestyle='--',
                        label=f'Лучший max_depth = {best_depth_f1}')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig('cv_vs_max_depth.png', dpi=150)
        plt.show()

        print(f"Лучший max_depth по Accuracy: {best_depth_acc} (Accuracy = {max(accuracy_scores):.4f})")
        print(f"Лучший max_depth по F1 Score: {best_depth_f1} (F1 = {max(f1_scores):.4f})")

        return best_depth_acc, accuracy_scores

    def cv_max_features(self, max_features_range=None, cv_folds=5):
        if max_features_range is None:
            max_features_range = range(1, self.X_scaled.shape[1] + 1)

        accuracy_scores = []

        for max_features in max_features_range:
            tree = DecisionTreeClassifier(max_features=max_features, random_state=42)
            scores = cross_val_score(tree, self.X_scaled, self.y,
                                     cv=cv_folds, scoring='accuracy')
            accuracy_scores.append(scores.mean())

        plt.figure(figsize=(10, 6))
        plt.plot(max_features_range, accuracy_scores, marker='s', linewidth=2, markersize=8, color='purple')
        plt.xlabel('max_features')
        plt.ylabel('Accuracy (Cross-Validation)')
        plt.title('Зависимость Accuracy от max_features')
        plt.grid(True, alpha=0.3)

        best_features = max_features_range[np.argmax(accuracy_scores)]
        plt.axvline(x=best_features, color='r', linestyle='--',
                    label=f'Лучший max_features = {best_features}')
        plt.legend()
        plt.tight_layout()
        plt.savefig('cv_vs_max_features.png', dpi=150)
        plt.show()

        print(f"Лучший max_features: {best_features} (Accuracy = {max(accuracy_scores):.4f})")
        return best_features, accuracy_scores

    def find_optimal_params(self):
        param_grid = {
            'max_depth': range(1, 21),
            'max_features': range(1, self.X_scaled.shape[1] + 1),
            'criterion': ['gini', 'entropy']
        }

        tree = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(
            tree, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )

        grid_search.fit(self.X_scaled, self.y)

        self.best_params = grid_search.best_params_
        self.tree_model = grid_search.best_estimator_

        print(f"Оптимальные параметры: {self.best_params}")
        print(f"Лучшая Accuracy: {grid_search.best_score_:.4f}")

        return self.best_params

    def plot_decision_boundaries(self):
        if self.X_scaled.shape[1] < 2:
            print("Нужно минимум 2 признака для границ")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        feature_pairs = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (2, 3)
        ]

        feature_names = self.column_names[:-1]

        for idx, (i, j) in enumerate(feature_pairs):
            if idx >= len(axes):
                break

            ax = axes[idx]

            try:
                display = DecisionBoundaryDisplay.from_estimator(
                    self.tree_model,
                    self.X_scaled[:, [i, j]],
                    response_method="predict",
                    ax=ax,
                    alpha=0.5,
                    grid_resolution=100
                )

                scatter = ax.scatter(
                    self.X_scaled[:, i],
                    self.X_scaled[:, j],
                    c=pd.factorize(self.y)[0],
                    edgecolor='black',
                    s=20,
                    alpha=0.7,
                    cmap='viridis'
                )

                ax.set_xlabel(feature_names[i])
                ax.set_ylabel(feature_names[j])
                ax.set_title(f'Границы: {feature_names[i]} vs {feature_names[j]}')
            except Exception as e:
                print(f"Ошибка при построении границ для {feature_names[i]}, {feature_names[j]}: {e}")
                ax.set_visible(False)

        plt.tight_layout()
        plt.savefig('decision_boundaries.png', dpi=150)
        plt.show()

    def feature_importance(self):
        if self.tree_model is None:
            print("Сначала обучите модель")
            return

        importance = self.tree_model.feature_importances_
        feature_names = self.column_names[:-1]

        sorted_idx = np.argsort(importance)[::-1]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(importance)), importance[sorted_idx], color='orange')
        plt.xticks(range(len(importance)), [feature_names[i] for i in sorted_idx], rotation=45)
        plt.xlabel('Признаки')
        plt.ylabel('Важность')
        plt.title('Важность признаков в дереве решений')

        for bar, imp in zip(bars, importance[sorted_idx]):
            if imp > 0:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{imp:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150)
        plt.show()

    def demo_classifier(self):
        if self.tree_model is None:
            print("Сначала обучите модель")
            return

        sample_data = {
            'Sex': 'M',
            'Length': 0.455,
            'Diameter': 0.365,
            'Height': 0.095,
            'Whole_weight': 0.5140,
            'Shucked_weight': 0.2245,
            'Viscera_weight': 0.1010,
            'Shell_weight': 0.150
        }

        df = pd.DataFrame([sample_data])
        df['Sex'] = self.label_encoder.transform(df['Sex'])
        df = df[self.column_names[:-1]]

        scaled_sample = self.scaler.transform(df)

        prediction = self.tree_model.predict(scaled_sample)[0]
        probabilities = self.tree_model.predict_proba(scaled_sample)[0]

        print("\n" + "=" * 50)
        print("ДЕМОНСТРАЦИЯ КЛАССИФИКАТОРА")
        print("=" * 50)
        print("\nВходные данные:")
        for key, value in sample_data.items():
            print(f"  {key}: {value}")

        print(f"\nПредсказанный класс: {prediction}")

        print("\nВероятности классов:")
        for class_name, prob in zip(self.tree_model.classes_, probabilities):
            print(f"  {class_name}: {prob:.4f}")

        print("\nПравила классификации:")
        print("  очень молодой: 0-5 колец (0-3.5 лет)")
        print("  молодой: 6-10 колец (4-8.5 лет)")
        print("  средний: 11-15 колец (9-13.5 лет)")
        print("  взрослый: 16-20 колец (14-18.5 лет)")
        print("  старый: 21+ колец (19+ лет)")


def main():
    print("=" * 60)
    print("ДЕРЕВО РЕШЕНИЙ ДЛЯ КЛАССИФИКАЦИИ ABALONE")
    print("=" * 60)

    model = AbaloneTreeModel('C:/Users/User/PycharmProjects/AI_prof_1/abalone.data')
    X, y = model.load_and_preprocess()

    print("\n" + "=" * 60)
    print("2.1 ПОСТРОЕНИЕ ЛОГИЧЕСКОГО КЛАССИФИКАТОРА")
    print("=" * 60)

    user_max_depth = int(input("Введите max_depth (например, 4): ") or "4")
    user_max_features = int(input(f"Введите max_features (1-{X.shape[1]}, например 4): ") or "4")

    model.build_tree(max_depth=user_max_depth, max_features=user_max_features)

    print("\n" + "=" * 60)
    print("2.2 ОЦЕНКА CROSS VALIDATION ДЛЯ max_depth")
    print("=" * 60)
    best_depth, depth_scores = model.cv_max_depth()

    print("\n" + "=" * 60)
    print("2.3 ОЦЕНКА CROSS VALIDATION ДЛЯ max_features")
    print("=" * 60)
    best_features, features_scores = model.cv_max_features()

    print("\n" + "=" * 60)
    print("2.4 ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ")
    print("=" * 60)
    optimal_params = model.find_optimal_params()

    print("\nОбоснование выбора:")
    print(f"1. max_depth={optimal_params['max_depth']} - оптимальная глубина")
    print("   обеспечивает баланс между точностью и переобучением")
    print(f"2. max_features={optimal_params['max_features']} - оптимальное количество")
    print("   признаков для разделения узлов")
    print(f"3. criterion={optimal_params['criterion']} - лучший критерий разделения")

    print("\n" + "=" * 60)
    print("2.5 ВИЗУАЛИЗАЦИЯ ДЕРЕВА")
    print("=" * 60)
    model.visualize_tree('optimal_decision_tree.png')

    print("\n" + "=" * 60)
    print("2.6 РЕШАЮЩИЕ ГРАНИЦЫ")
    print("=" * 60)
    model.plot_decision_boundaries()

    print("\n" + "=" * 60)
    print("АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
    print("=" * 60)
    model.feature_importance()

    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ КЛАССИФИКАТОРА")
    print("=" * 60)
    model.demo_classifier()

    print("\n" + "=" * 60)
    print("СОХРАНЕННЫЕ ФАЙЛЫ:")
    print("=" * 60)
    print("1. optimal_decision_tree.png - визуализация дерева")
    print("2. cv_vs_max_depth.png - график зависимости от глубины")
    print("3. cv_vs_max_features.png - график зависимости от признаков")
    print("4. decision_boundaries.png - решающие границы")
    print("5. feature_importance.png - важность признаков")


if __name__ == "__main__":
    main()