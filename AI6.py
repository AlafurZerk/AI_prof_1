import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class AbaloneRegressionPipeline:
    def __init__(self, file_path='abalone.data'):
        self.file_path = file_path
        self.column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
                             'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.preprocessor = None

    def load_and_prepare_data(self):
        self.df = pd.read_csv(self.file_path, names=self.column_names)

        X = self.df.drop('Rings', axis=1)
        y = self.df['Rings']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def create_preprocessing_pipeline(self):
        numeric_features = ['Length', 'Diameter', 'Height', 'Whole_weight',
                            'Shucked_weight', 'Viscera_weight', 'Shell_weight']
        categorical_features = ['Sex']

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        return self.preprocessor

    def create_regression_models(self):
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Lasso Regression': Lasso(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'SVR': SVR()
        }

        return models

    def evaluate_model(self, model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"{model_name}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R2 Score: {r2:.4f}")
        print("-" * 40)

        return {
            'model_name': model_name,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }

    def train_and_evaluate_models(self):
        self.load_and_prepare_data()
        self.create_preprocessing_pipeline()
        models = self.create_regression_models()

        results = []

        for name, model in models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('regressor', model)
            ])

            pipeline.fit(self.X_train, self.y_train)

            result = self.evaluate_model(pipeline, self.X_test, self.y_test, name)
            result['model'] = pipeline
            results.append(result)

        self.results_df = pd.DataFrame(results)
        self.best_model = self.results_df.loc[self.results_df['r2'].idxmax(), 'model']

        return results

    def hyperparameter_tuning(self):
        base_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])

        param_grid = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [10, 20, 30, None],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(
            base_pipeline,
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.X_train, self.y_train)

        print(f"\nЛучшие параметры: {grid_search.best_params_}")
        print(f"Лучший R2 score: {grid_search.best_score_:.4f}")

        self.best_model = grid_search.best_estimator_

        y_pred = self.best_model.predict(self.X_test)
        final_r2 = r2_score(self.y_test, y_pred)
        print(f"R2 score на тестовой выборке: {final_r2:.4f}")

        return grid_search.best_estimator_

    def cross_validation(self, model, cv=5):
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', model)
        ])

        cv_scores = cross_val_score(
            pipeline, self.X_train, self.y_train,
            cv=cv, scoring='r2', n_jobs=-1
        )

        print(f"\nКросс-валидация (R2 score):")
        print(f"  Среднее: {cv_scores.mean():.4f}")
        print(f"  Стандартное отклонение: {cv_scores.std():.4f}")
        print(f"  Все оценки: {cv_scores}")

        return cv_scores

    def predict_new_data(self, new_data):
        if self.best_model is None:
            raise ValueError("Сначала обучите модель!")

        if isinstance(new_data, pd.DataFrame):
            predictions = self.best_model.predict(new_data)
        else:
            new_df = pd.DataFrame([new_data], columns=self.column_names[:-1])
            predictions = self.best_model.predict(new_df)

        return predictions

    def feature_importance_analysis(self):
        if not hasattr(self.best_model.named_steps['regressor'], 'feature_importances_'):
            print("Данная модель не поддерживает анализ важности признаков")
            return None

        preprocessor = self.best_model.named_steps['preprocessor']
        feature_names = []

        for name, transformer, columns in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                encoder = transformer.named_steps['onehot']
                cat_features = encoder.get_feature_names_out(columns)
                feature_names.extend(cat_features)

        importances = self.best_model.named_steps['regressor'].feature_importances_

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\nВажность признаков:")
        print(importance_df)

        return importance_df

    def run_complete_pipeline(self, use_hyperparameter_tuning=True):
        print("=" * 60)
        print("ЗАПУСК УНИВЕРСАЛЬНОГО ПАЙПЛАЙНА РЕГРЕССИИ")
        print("=" * 60)

        print("\n1. Загрузка и подготовка данных...")
        self.load_and_prepare_data()
        print(f"   Размер обучающей выборки: {self.X_train.shape}")
        print(f"   Размер тестовой выборки: {self.X_test.shape}")

        print("\n2. Создание пайплайна предобработки...")
        self.create_preprocessing_pipeline()

        print("\n3. Обучение и оценка базовых моделей...")
        results = self.train_and_evaluate_models()

        print("\n4. Результаты сравнения моделей:")
        print(self.results_df[['model_name', 'r2', 'mse', 'mae']].sort_values('r2', ascending=False))

        if use_hyperparameter_tuning:
            print("\n5. Настройка гиперпараметров лучшей модели...")
            self.hyperparameter_tuning()

        print("\n6. Кросс-валидация лучшей модели...")
        best_regressor = self.best_model.named_steps['regressor']
        self.cross_validation(best_regressor)

        print("\n7. Анализ важности признаков...")
        self.feature_importance_analysis()

        print("\n" + "=" * 60)
        print("ПАЙПЛАЙН УСПЕШНО ВЫПОЛНЕН")
        print("=" * 60)

        return self.best_model


if __name__ == "__main__":
    pipeline = AbaloneRegressionPipeline('abalone.data')
    best_model = pipeline.run_complete_pipeline(use_hyperparameter_tuning=True)

    print(f"\nЛучшая модель: {type(best_model.named_steps['regressor']).__name__}")

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

    prediction = pipeline.predict_new_data(sample_data)
    print(f"\nПредсказание для нового образца: {prediction[0]:.1f} колец")
    print(f"Примерный возраст: {prediction[0] + 1.5:.1f} лет")