import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class AbaloneDataPreprocessor:
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

    def load_data(self):
        self.data = pd.read_csv(self.data_path, names=self.column_names)
        print(f"Данные загружены. Размер: {self.data.shape}")

    def preprocess(self, problem_type='regression', n_bins=10):
        if self.data is None:
            raise ValueError("Сначала загрузите данные")

        self.data['Sex'] = self.label_encoder.fit_transform(self.data['Sex'])
        self.X = self.data.drop('Rings', axis=1)

        if problem_type == 'regression':
            self.y = self.data['Rings'].values
            print(f"Режим: регрессия")
        elif problem_type == 'classification':
            self.y = pd.cut(self.data['Rings'], bins=n_bins, labels=False).values
            print(f"Режим: классификация, {n_bins} классов")
        else:
            raise ValueError("Выберите 'regression' или 'classification'")

        self.X_scaled = self.scaler.fit_transform(self.X)
        print("Данные готовы")

    def get_data(self):
        if self.X_scaled is None or self.y is None:
            raise ValueError("Сначала предобработайте данные")
        return self.X_scaled, self.y

    def get_feature_names(self):
        return self.column_names[:-1]


class KNNExperiment:
    def __init__(self, X, y, problem_type='regression'):
        self.X = X
        self.y = y
        self.problem_type = problem_type
        self.results_hold_out = {}
        self.results_cv = {}
        self.optimal_k = None
        self.best_model = None

    def create_model(self, k):
        if self.problem_type == 'regression':
            return KNeighborsRegressor(n_neighbors=k)
        else:
            return KNeighborsClassifier(n_neighbors=k)

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)

        if self.problem_type == 'regression':
            return {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        else:
            return {'accuracy': accuracy_score(y_test, y_pred)}

    def run_hold_out_evaluation(self, test_sizes=None, k_values=None, random_state=42):
        if test_sizes is None:
            test_sizes = [0.2, 0.3, 0.4]
        if k_values is None:
            k_values = range(1, 31)

        self.results_hold_out = {}

        for test_size in test_sizes:
            self.results_hold_out[test_size] = {}
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state
            )

            for k in k_values:
                model = self.create_model(k)
                model.fit(X_train, y_train)
                metrics = self.evaluate_model(model, X_test, y_test)

                if self.problem_type == 'regression':
                    self.results_hold_out[test_size][k] = -metrics['mse']
                else:
                    self.results_hold_out[test_size][k] = metrics['accuracy']

        return self.results_hold_out

    def run_cross_validation_evaluation(self, folds=None, k_values=None):
        if folds is None:
            folds = [3, 5, 7]
        if k_values is None:
            k_values = range(1, 31)

        self.results_cv = {}

        for fold in folds:
            self.results_cv[fold] = {}

            for k in k_values:
                model = self.create_model(k)
                kf = KFold(n_splits=fold, shuffle=True, random_state=42)

                if self.problem_type == 'regression':
                    cv_scores = cross_val_score(model, self.X, self.y, cv=kf,
                                                scoring='neg_mean_squared_error')
                    self.results_cv[fold][k] = np.mean(cv_scores)
                else:
                    cv_scores = cross_val_score(model, self.X, self.y, cv=kf, scoring='accuracy')
                    self.results_cv[fold][k] = np.mean(cv_scores)

        return self.results_cv

    def find_optimal_k(self, method='cv'):
        if method == 'hold_out':
            avg_scores = {}
            for k in range(1, 31):
                scores = []
                for test_size in self.results_hold_out:
                    if k in self.results_hold_out[test_size]:
                        scores.append(self.results_hold_out[test_size][k])
                if scores:
                    avg_scores[k] = np.mean(scores)

            if avg_scores:
                self.optimal_k = max(avg_scores, key=avg_scores.get)
                print(f"Оптимальное k (hold-out): {self.optimal_k}")

        elif method == 'cv':
            avg_scores = {}
            for k in range(1, 31):
                scores = []
                for fold in self.results_cv:
                    if k in self.results_cv[fold]:
                        scores.append(self.results_cv[fold][k])
                if scores:
                    avg_scores[k] = np.mean(scores)

            if avg_scores:
                self.optimal_k = max(avg_scores, key=avg_scores.get)
                print(f"Оптимальное k (cross-validation): {self.optimal_k}")

        return self.optimal_k

    def train_final_model(self, k=None, test_size=0.3, random_state=42):
        if k is None:
            if self.optimal_k is None:
                self.find_optimal_k('cv')
            k = self.optimal_k

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        self.best_model = self.create_model(k)
        self.best_model.fit(X_train, y_train)

        metrics = self.evaluate_model(self.best_model, X_test, y_test)

        print(f"Модель обучена с k={k}")

        if self.problem_type == 'regression':
            print(f"MSE: {metrics['mse']:.4f}")
            print(f"MAE: {metrics['mae']:.4f}")
            print(f"R²: {metrics['r2']:.4f}")
        else:
            print(f"Accuracy: {metrics['accuracy']:.4f}")

        return self.best_model

    def plot_results(self):
        if not self.results_hold_out and not self.results_cv:
            print("Нет данных для графиков")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        if self.results_hold_out:
            ax1 = axes[0]
            for test_size, scores in self.results_hold_out.items():
                ax1.plot(list(scores.keys()), list(scores.values()),
                         label=f'test_size={test_size}', marker='o')
            ax1.set_xlabel('k')

            if self.problem_type == 'regression':
                ax1.set_ylabel('-MSE')
                ax1.set_title('Hold-out (регрессия)')
            else:
                ax1.set_ylabel('Accuracy')
                ax1.set_title('Hold-out (классификация)')

            ax1.legend()
            ax1.grid(True)

        if self.results_cv:
            ax2 = axes[1]
            for fold, scores in self.results_cv.items():
                ax2.plot(list(scores.keys()), list(scores.values()),
                         label=f'folds={fold}', marker='s')
            ax2.set_xlabel('k')

            if self.problem_type == 'regression':
                ax2.set_ylabel('-MSE')
                ax2.set_title('Cross-validation (регрессия)')
            else:
                ax2.set_ylabel('Accuracy')
                ax2.set_title('Cross-validation (классификация)')

            ax2.legend()
            ax2.grid(True)

        plt.tight_layout()
        plt.show()


class AbalonePredictorDemo:
    def __init__(self, model, scaler, label_encoder, feature_names, problem_type='regression'):
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.feature_names = feature_names
        self.problem_type = problem_type

    def predict_sample(self, sample_data):
        df = pd.DataFrame([sample_data])

        if 'Sex' in df.columns:
            df['Sex'] = self.label_encoder.transform(df['Sex'])

        df = df[self.feature_names]
        scaled_sample = self.scaler.transform(df)
        prediction = self.model.predict(scaled_sample)

        return prediction[0]


class AbaloneAnalysisSystem:
    def __init__(self, data_path='abalone.data'):
        self.data_path = data_path
        self.preprocessor = None
        self.experiment = None
        self.demo = None

    def run_analysis(self, problem_type='regression', n_bins=10):
        print("=" * 60)
        print(f"АНАЛИЗ ABALONE - {problem_type.upper()}")
        print("=" * 60)

        print("\n1. Загрузка данных")
        self.preprocessor = AbaloneDataPreprocessor(self.data_path)
        self.preprocessor.load_data()
        self.preprocessor.preprocess(problem_type=problem_type, n_bins=n_bins)
        X, y = self.preprocessor.get_data()

        print("\n2. Создание эксперимента")
        self.experiment = KNNExperiment(X, y, problem_type=problem_type)

        print("\n3. Hold-out оценка")
        self.experiment.run_hold_out_evaluation()

        print("\n4. Cross-validation оценка")
        self.experiment.run_cross_validation_evaluation()

        print("\n5. Поиск оптимального k")
        self.experiment.find_optimal_k('cv')

        print("\n6. Графики")
        self.experiment.plot_results()

        print("\n7. Обучение финальной модели")
        final_model = self.experiment.train_final_model()

        print("\n8. Демонстрация")
        self.demo = AbalonePredictorDemo(
            model=final_model,
            scaler=self.preprocessor.scaler,
            label_encoder=self.preprocessor.label_encoder,
            feature_names=self.preprocessor.get_feature_names(),
            problem_type=problem_type
        )

        sample = {
            'Sex': 'M',
            'Length': 0.455,
            'Diameter': 0.365,
            'Height': 0.095,
            'Whole_weight': 0.5140,
            'Shucked_weight': 0.2245,
            'Viscera_weight': 0.1010,
            'Shell_weight': 0.150
        }

        print("\nПример предсказания:")
        for key, value in sample.items():
            print(f"  {key}: {value}")

        prediction = self.demo.predict_sample(sample)

        if problem_type == 'regression':
            age_pred = prediction + 1.5
            print(f"Предсказанные кольца: {prediction:.2f}")
            print(f"Предсказанный возраст: {age_pred:.2f} лет")
        else:
            print(f"Предсказанный класс: {int(prediction)}")

        print("\n" + "=" * 60)
        print("ГОТОВО")
        print("=" * 60)

        return self.experiment.best_model


if __name__ == "__main__":
    system = AbaloneAnalysisSystem('abalone.data')

    print("Выберите режим:")
    print("1. Регрессия (предсказание колец)")
    print("2. Классификация (группировка в классы)")

    choice = input("Введите 1 или 2: ")

    if choice == '1':
        best_model = system.run_analysis(problem_type='regression')
    else:
        n_bins = int(input("Количество классов (5-10): "))
        best_model = system.run_analysis(problem_type='classification', n_bins=n_bins)

    print("\n\n" + "=" * 60)
    print("ДОПОЛНИТЕЛЬНАЯ ДЕМОНСТРАЦИЯ")
    print("=" * 60)

    preprocessor = AbaloneDataPreprocessor('abalone.data')
    preprocessor.load_data()
    preprocessor.preprocess(problem_type='regression')
    X, y = preprocessor.get_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    user_k = int(input("Введите k для модели: "))

    user_model = KNeighborsRegressor(n_neighbors=user_k)
    user_model.fit(X_train, y_train)

    y_pred = user_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\nМодель с k={user_k}:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Ошибка в возрасте: {mae * 1.5:.2f} лет")

    demo_user = AbalonePredictorDemo(
        model=user_model,
        scaler=preprocessor.scaler,
        label_encoder=preprocessor.label_encoder,
        feature_names=preprocessor.get_feature_names(),
        problem_type='regression'
    )

    sample = {
        'Sex': 'F',
        'Length': 0.350,
        'Diameter': 0.265,
        'Height': 0.090,
        'Whole_weight': 0.2255,
        'Shucked_weight': 0.0995,
        'Viscera_weight': 0.0485,
        'Shell_weight': 0.070
    }

    prediction = demo_user.predict_sample(sample)
    print(f"\nПредсказание для нового образца: {prediction:.2f} колец")
    print(f"Возраст: {prediction + 1.5:.2f} лет")