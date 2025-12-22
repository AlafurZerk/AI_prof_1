import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class AbaloneDataProcessor:
    def __init__(self, file_path='abalone.data'):
        self.file_path = file_path
        self.column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
                             'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path, names=self.column_names)
        return self.df

    def explore_data(self):
        print("ИНФОРМАЦИЯ О ДАТАФРЕЙМЕ:")
        print(self.df.info())
        print("\nПЕРВЫЕ 5 СТРОК ДАННЫХ:")
        print(self.df.head())
        print("\nСТАТИСТИЧЕСКОЕ ОПИСАНИЕ:")
        print(self.df.describe())
        missing_values = self.df.isnull().sum()
        print("\nПРОВЕРКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ:")
        print(missing_values[missing_values > 0] if missing_values.any() else "Пропущенных значений нет")
        print(f"\nКОЛИЧЕСТВО ДУБЛИКАТОВ: {self.df.duplicated().sum()}")

    def visualize_data(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(4, 3, figsize=(15, 16))
        fig.suptitle('Распределение признаков датасета Abalone', fontsize=16, y=1.02)

        sns.countplot(data=self.df, x='Sex', ax=axes[0, 0], hue='Sex', palette='Set2', legend=False)
        axes[0, 0].set_title('Распределение по полу')

        sns.histplot(self.df['Rings'], bins=30, ax=axes[0, 1], kde=True, color='skyblue')
        axes[0, 1].set_title('Распределение колец')
        axes[0, 1].set_xlabel('Количество колец')
        axes[0, 1].set_ylabel('Частота')

        self.df['Age'] = self.df['Rings'] + 1.5
        sns.histplot(self.df['Age'], bins=30, ax=axes[0, 2], kde=True, color='lightcoral')
        axes[0, 2].set_title('Распределение возраста')
        axes[0, 2].set_xlabel('Возраст (годы)')
        axes[0, 2].set_ylabel('Частота')

        numeric_features = ['Length', 'Diameter', 'Height', 'Whole_weight',
                            'Shucked_weight', 'Viscera_weight', 'Shell_weight']

        for idx, feature in enumerate(numeric_features):
            row = (idx + 3) // 3
            col = (idx + 3) % 3
            sns.histplot(self.df[feature], bins=30, ax=axes[row, col], kde=True)
            axes[row, col].set_title(f'Распределение {feature}')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Частота')

        for i in range(len(numeric_features) + 3, 12):
            row = i // 3
            col = i % 3
            axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 8))
        correlation_matrix = self.df[numeric_features + ['Rings']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Матрица корреляций числовых признаков')
        plt.tight_layout()
        plt.show()

        if 'Age' in self.df.columns:
            self.df = self.df.drop('Age', axis=1)

    def encode_categorical_data(self, method='onehot'):
        if method == 'onehot':
            self.df = pd.get_dummies(self.df, columns=['Sex'], prefix='Sex', drop_first=False)
        elif method == 'label':
            le = LabelEncoder()
            self.df['Sex_encoded'] = le.fit_transform(self.df['Sex'])
            self.df = self.df.drop('Sex', axis=1)

        print(f"Размерность данных после кодирования: {self.df.shape}")
        return self.df

    def split_data(self, test_size=0.2, random_state=42):
        X = self.df.drop('Rings', axis=1)
        y = self.df['Rings']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"Тренировочная выборка: {self.X_train.shape}")
        print(f"Тестовая выборка: {self.X_test.shape}")
        print(f"Соотношение: {len(self.X_train) / len(X):.1%}/{len(self.X_test) / len(X):.1%}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale_features(self):
        numeric_features = ['Length', 'Diameter', 'Height', 'Whole_weight',
                            'Shucked_weight', 'Viscera_weight', 'Shell_weight']

        scaler = StandardScaler()

        self.X_train_scaled = self.X_train.copy()
        self.X_test_scaled = self.X_test.copy()

        self.X_train_scaled[numeric_features] = scaler.fit_transform(self.X_train[numeric_features])
        self.X_test_scaled[numeric_features] = scaler.transform(self.X_test[numeric_features])

        print("Масштабирование признаков выполнено")
        return self.X_train_scaled, self.X_test_scaled

    def process_pipeline(self, encode_method='onehot', test_size=0.2, random_state=42):
        self.load_data()
        self.explore_data()
        self.visualize_data()
        self.encode_categorical_data(method=encode_method)
        self.split_data(test_size=test_size, random_state=random_state)
        self.scale_features()
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test


class AbaloneAnalysis:
    def __init__(self, processor):
        self.processor = processor

    def analyze_target_distribution(self):
        rings_counts = self.processor.df['Rings'].value_counts().sort_index()
        plt.figure(figsize=(12, 6))
        rings_counts.plot(kind='bar')
        plt.title('Распределение целевой переменной (Rings)')
        plt.xlabel('Количество колец')
        plt.ylabel('Частота')
        plt.tight_layout()
        plt.show()

        print("Статистика по целевой переменной:")
        print(self.processor.df['Rings'].describe())

    def analyze_feature_relationships(self):
        numeric_features = ['Length', 'Diameter', 'Height', 'Whole_weight',
                            'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()

        for idx, feature in enumerate(numeric_features[:-1]):
            axes[idx].scatter(self.processor.df[feature], self.processor.df['Rings'], alpha=0.5)
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Rings')
            axes[idx].set_title(f'{feature} vs Rings')

        plt.tight_layout()
        plt.show()

    def run_full_analysis(self):
        self.analyze_target_distribution()
        self.analyze_feature_relationships()


if __name__ == "__main__":
    processor = AbaloneDataProcessor('abalone.data')
    X_train_scaled, X_test_scaled, y_train, y_test = processor.process_pipeline()

    analysis = AbaloneAnalysis(processor)
    analysis.run_full_analysis()

    processor.df.to_csv('abalone_processed.csv', index=False)
    print("Обработанные данные сохранены в 'abalone_processed.csv'")