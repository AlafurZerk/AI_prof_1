import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class BackwardEliminationRegressor:
    def __init__(self, significance_level=0.05):
        self.significance_level = significance_level
        self.selected_features = []
        self.model = None
        self.encoder = None

    def prepare_data(self, X):
        if isinstance(X, pd.DataFrame):
            X_copy = X.copy()
        else:
            X_copy = pd.DataFrame(X)

        categorical_cols = X_copy.select_dtypes(include=['object', 'category']).columns
        numeric_cols = X_copy.select_dtypes(include=[np.number]).columns

        X_numeric = X_copy[numeric_cols].reset_index(drop=True)

        if len(categorical_cols) > 0:
            if self.encoder is None:
                self.encoder = OneHotEncoder(drop='first', sparse_output=False)
                encoded_data = self.encoder.fit_transform(X_copy[categorical_cols])
            else:
                try:
                    encoded_data = self.encoder.transform(X_copy[categorical_cols])
                except:
                    encoded_data = self.encoder.fit_transform(X_copy[categorical_cols])

            encoded_df = pd.DataFrame(
                encoded_data,
                columns=self.encoder.get_feature_names_out(categorical_cols)
            ).reset_index(drop=True)

            X_processed = pd.concat([X_numeric, encoded_df], axis=1)
        else:
            X_processed = X_numeric

        X_processed = sm.add_constant(X_processed, has_constant='add')
        return X_processed

    def backward_elimination(self, X, y):
        X_processed = self.prepare_data(X)

        y_series = pd.Series(y).reset_index(drop=True)
        X_processed = X_processed.reset_index(drop=True)

        num_features = X_processed.shape[1]

        for i in range(num_features):
            if X_processed.shape[1] <= 1:
                break

            model = sm.OLS(y_series, X_processed).fit()
            p_values = model.pvalues
            max_p_value = p_values.max()

            if max_p_value > self.significance_level:
                excluded_feature = p_values.idxmax()
                if excluded_feature == 'const':
                    if X_processed.shape[1] > 1:
                        excluded_feature = p_values[p_values.index != 'const'].idxmax()
                    else:
                        break
                X_processed = X_processed.drop(excluded_feature, axis=1)
            else:
                break

        self.model = sm.OLS(y_series, X_processed).fit()
        self.selected_features = X_processed.columns.tolist()

        return self.model

    def fit(self, X, y):
        return self.backward_elimination(X, y)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Модель не обучена!")

        X_processed = self.prepare_data(X)
        X_processed = X_processed[self.selected_features]

        return self.model.predict(X_processed)

    def get_selected_features(self):
        return [feat for feat in self.selected_features if feat != 'const']

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_test_series = pd.Series(y_test).reset_index(drop=True)
        y_pred_series = pd.Series(y_pred).reset_index(drop=True)

        mse = mean_squared_error(y_test_series, y_pred_series)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_series, y_pred_series)
        r2 = r2_score(y_test_series, y_pred_series)

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

        return metrics, y_pred


class AbaloneBackwardElimination:
    def __init__(self, file_path='abalone.data'):
        self.file_path = file_path
        self.column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
                             'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def load_and_prepare_data(self):
        self.df = pd.read_csv(self.file_path, names=self.column_names)

        X = self.df.drop('Rings', axis=1)
        y = self.df['Rings']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def run_backward_elimination(self, significance_level=0.05):
        self.load_and_prepare_data()

        print("=" * 60)
        print("МОДЕЛЬ МНОГОМЕРНОЙ РЕГРЕССИИ С BACKWARD ELIMINATION")
        print("=" * 60)

        model = BackwardEliminationRegressor(significance_level=significance_level)

        print("\n1. Обучение модели с Backward Elimination...")
        fitted_model = model.fit(self.X_train, self.y_train)

        print("\n2. Отобранные признаки:")
        selected_features = model.get_selected_features()
        for i, feature in enumerate(selected_features, 1):
            print(f"   {i}. {feature}")

        print(f"\n3. Итоговое количество признаков: {len(selected_features)}")

        print("\n4. Ключевая статистика модели:")
        print(f"   R-squared: {fitted_model.rsquared:.4f}")
        print(f"   Adjusted R-squared: {fitted_model.rsquared_adj:.4f}")
        print(f"   F-statistic: {fitted_model.fvalue:.4f}")
        print(f"   Prob (F-statistic): {fitted_model.f_pvalue:.4f}")

        print("\n5. Коэффициенты модели:")
        print(fitted_model.params)

        print("\n6. Оценка на тестовой выборке:")
        metrics, y_pred = model.evaluate(self.X_test, self.y_test)

        for metric_name, value in metrics.items():
            print(f"   {metric_name}: {value:.4f}")

        self.model = model
        return model, metrics

    def visualize_results(self):
        if self.model is None:
            print("Сначала обучите модель!")
            return

        metrics, y_pred = self.model.evaluate(self.X_test, self.y_test)
        y_test_series = pd.Series(self.y_test).reset_index(drop=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].scatter(y_test_series, y_pred, alpha=0.5, color='blue')
        axes[0].plot([y_test_series.min(), y_test_series.max()],
                     [y_test_series.min(), y_test_series.max()],
                     'r--', lw=2)
        axes[0].set_xlabel('Фактические значения (Rings)')
        axes[0].set_ylabel('Предсказанные значения (Rings)')
        axes[0].set_title('Фактические vs Предсказанные значения')
        axes[0].grid(True, alpha=0.3)

        residuals = y_test_series.values - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, color='green')
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Предсказанные значения')
        axes[1].set_ylabel('Остатки')
        axes[1].set_title('Остатки модели')
        axes[1].grid(True, alpha=0.3)

        axes[2].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[2].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[2].set_xlabel('Остатки')
        axes[2].set_ylabel('Частота')
        axes[2].set_title('Распределение остатков')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        residuals_stats = pd.Series(residuals).describe()
        print("\nСтатистика остатков:")
        print(residuals_stats)

    def compare_with_full_model(self):
        X_train_encoded = pd.get_dummies(self.X_train, columns=['Sex'], drop_first=True)
        X_test_encoded = pd.get_dummies(self.X_test, columns=['Sex'], drop_first=True)

        X_train_const = sm.add_constant(X_train_encoded, has_constant='add')
        X_test_const = sm.add_constant(X_test_encoded, has_constant='add')

        full_model = sm.OLS(self.y_train, X_train_const).fit()
        full_pred = full_model.predict(X_test_const)

        full_metrics = {
            'MSE': mean_squared_error(self.y_test, full_pred),
            'RMSE': np.sqrt(mean_squared_error(self.y_test, full_pred)),
            'MAE': mean_absolute_error(self.y_test, full_pred),
            'R2': r2_score(self.y_test, full_pred)
        }

        backward_metrics, _ = self.model.evaluate(self.X_test, self.y_test)

        comparison_df = pd.DataFrame({
            'Полная модель': full_metrics,
            'Backward Elimination': backward_metrics
        })

        print("\n" + "=" * 60)
        print("СРАВНЕНИЕ С ПОЛНОЙ МОДЕЛЬЮ")
        print("=" * 60)
        print(comparison_df)

        print("\nПолная модель R-squared:", full_model.rsquared)
        print("Backward Elimination R-squared:", self.model.model.rsquared)

        return comparison_df

    def predict_new_sample(self, sample_data):
        if self.model is None:
            print("Сначала обучите модель!")
            return

        if isinstance(sample_data, dict):
            sample_df = pd.DataFrame([sample_data])
        else:
            sample_df = sample_data

        prediction = self.model.predict(sample_df)
        age_prediction = prediction[0] + 1.5

        print(f"\nПредсказание для нового образца:")
        print(f"  Количество колец: {prediction[0]:.1f}")
        print(f"  Примерный возраст: {age_prediction:.1f} лет")

        return prediction[0], age_prediction

    def feature_importance_analysis(self):
        if self.model is None:
            print("Сначала обучите модель!")
            return

        coefficients = self.model.model.params
        std_errors = self.model.model.bse
        t_values = self.model.model.tvalues
        p_values = self.model.model.pvalues

        importance_df = pd.DataFrame({
            'Коэффициент': coefficients,
            'Стандартная ошибка': std_errors,
            't-статистика': t_values,
            'p-значение': p_values
        })

        print("\n" + "=" * 60)
        print("АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
        print("=" * 60)
        print(importance_df.sort_values('p-значение'))

        return importance_df


if __name__ == "__main__":
    abalone_model = AbaloneBackwardElimination('abalone.data')

    model, metrics = abalone_model.run_backward_elimination(significance_level=0.05)

    abalone_model.visualize_results()

    abalone_model.feature_importance_analysis()

    abalone_model.compare_with_full_model()

    sample_data = {
        'Sex': 'M',
        'Length': 0.455,
        'Diameter': 0.365,
        'Height': 0.095,
        'Whole_weight': 0.514,
        'Shucked_weight': 0.2245,
        'Viscera_weight': 0.101,
        'Shell_weight': 0.15
    }

    abalone_model.predict_new_sample(sample_data)