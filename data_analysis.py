import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Настройка отображения графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Загрузка данных
columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
           'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
df = pd.read_csv('abalone.data', names=columns)

print("=" * 60)
print("ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ ABALONE")
print("=" * 60)

# Основная информация о данных
print("\n1. ОСНОВНАЯ ИНФОРМАЦИЯ О ДАННЫХ")
print("-" * 40)
print(f"Размерность данных: {df.shape}")
print(f"Количество записей: {df.shape[0]}")
print(f"Количество признаков: {df.shape[1]}")

# Проверка пропущенных значений
print("\n2. ПРОВЕРКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
print("-" * 40)
missing_values = df.isnull().sum()
print(missing_values)

# Статистика по числовым признакам
print("\n3. СТАТИСТИКА ПО ЧИСЛОВЫМ ПРИЗНАКАМ")
print("-" * 40)
print(df.describe())

# Статистика по категориальному признаку
print("\n4. СТАТИСТИКА ПО ПРИЗНАКУ 'Sex'")
print("-" * 40)
sex_counts = df['Sex'].value_counts()
sex_percent = df['Sex'].value_counts(normalize=True) * 100
sex_stats = pd.DataFrame({
    'Count': sex_counts,
    'Percentage': sex_percent
})
print(sex_stats)

# Статистика по целевой переменной
print("\n5. СТАТИСТИКА ПО ЦЕЛЕВОЙ ПЕРЕМЕННОЙ 'Rings'")
print("-" * 40)
print(f"Минимальное количество колец: {df['Rings'].min()}")
print(f"Максимальное количество колец: {df['Rings'].max()}")
print(f"Среднее количество колец: {df['Rings'].mean():.2f}")
print(f"Медианное количество колец: {df['Rings'].median()}")
print(f"Стандартное отклонение: {df['Rings'].std():.2f}")

# Расчет возраста
df['Age'] = df['Rings'] + 1.5
print(f"\nСоответствующий возраст (Rings + 1.5):")
print(f"Минимальный возраст: {df['Age'].min():.1f} лет")
print(f"Максимальный возраст: {df['Age'].max():.1f} лет")
print(f"Средний возраст: {df['Age'].mean():.1f} лет")

# ВИЗУАЛИЗАЦИИ
print("\n6. ПОСТРОЕНИЕ ГРАФИЧЕСКИХ ПРЕДСТАВЛЕНИЙ")
print("-" * 40)

# 6.1 Распределение целевой переменной
plt.figure(figsize=(15, 12))

plt.subplot(3, 3, 1)
sns.histplot(data=df, x='Rings', bins=30, kde=True)
plt.title('Распределение количества колец (Rings)')
plt.xlabel('Количество колец')
plt.ylabel('Частота')

# 6.2 Распределение по полу
plt.subplot(3, 3, 2)
sns.countplot(data=df, x='Sex', order=['M', 'F', 'I'])
plt.title('Распределение по полу')
plt.xlabel('Пол')
plt.ylabel('Количество')

# 6.3 Boxplot колец по полу
plt.subplot(3, 3, 3)
sns.boxplot(data=df, x='Sex', y='Rings', order=['M', 'F', 'I'])
plt.title('Распределение колец по полу')
plt.xlabel('Пол')
plt.ylabel('Количество колец')

# 6.4 Матрица корреляций
plt.subplot(3, 3, 4)
numeric_df = df.select_dtypes(include=[np.number])
# Исключаем Age, так как он линейно зависит от Rings
numeric_df_for_corr = numeric_df.drop('Age', axis=1, errors='ignore')
correlation_matrix = numeric_df_for_corr.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
            center=0, square=True, fmt='.2f')
plt.title('Матрица корреляций (числовые признаки)')
plt.tight_layout()

# 6.5 Выбросы в физических размерах
plt.subplot(3, 3, 5)
sns.boxplot(data=df[['Length', 'Diameter', 'Height']])
plt.title('Распределение физических размеров')
plt.ylabel('Значение (после масштабирования)')
plt.xticks(rotation=45)

# 6.6 Выбросы в весах
plt.subplot(3, 3, 6)
sns.boxplot(data=df[['Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']])
plt.title('Распределение весов')
plt.ylabel('Значение (после масштабирования)')
plt.xticks(rotation=45)

# 6.7 Scatter plot: Shell_weight vs Rings
plt.subplot(3, 3, 7)
sns.scatterplot(data=df, x='Shell_weight', y='Rings', hue='Sex', alpha=0.6)
plt.title('Shell_weight vs Rings')
plt.xlabel('Вес раковины')
plt.ylabel('Количество колец')

# 6.8 Scatter plot: Diameter vs Length
plt.subplot(3, 3, 8)
sns.scatterplot(data=df, x='Length', y='Diameter', hue='Sex', alpha=0.6)
plt.title('Diameter vs Length')
plt.xlabel('Длина')
plt.ylabel('Диаметр')

# 6.9 Распределение Height с выбросами
plt.subplot(3, 3, 9)
sns.boxplot(data=df, y='Height')
plt.title('Распределение Height (выбросы)')
plt.ylabel('Высота')

plt.tight_layout()
plt.show()

# ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ
print("\n7. ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ")
print("-" * 40)

# Анализ выбросов в Height
Q1 = df['Height'].quantile(0.25)
Q3 = df['Height'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_height = df[(df['Height'] < lower_bound) | (df['Height'] > upper_bound)]
print(f"Выбросы в признаке Height: {len(outliers_height)} записей")
print(f"Верхняя граница для выбросов: {upper_bound:.3f}")
print(f"Максимальное значение Height: {df['Height'].max():.3f}")

# Корреляция с целевой переменной
correlation_with_rings = numeric_df_for_corr.corr()['Rings'].sort_values(ascending=False)
print("\nКорреляция признаков с целевой переменной (Rings):")
for feature, corr in correlation_with_rings.items():
    if feature != 'Rings':
        print(f"  {feature}: {corr:.3f}")

# Анализ по возрастным группам
df['Age_Group'] = pd.cut(df['Rings'],
                         bins=[0, 8, 11, 30],
                         labels=['Молодые (1-8)', 'Взрослые (9-11)', 'Старые (12+)'])

print(f"\nРаспределение по возрастным группам:")
print(df['Age_Group'].value_counts())

# Визуализация возрастных групп
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.countplot(data=df, x='Age_Group')
plt.title('Распределение по возрастным группам')
plt.xlabel('Возрастная группа')
plt.ylabel('Количество')

plt.subplot(1, 3, 2)
sns.boxplot(data=df, x='Age_Group', y='Shell_weight')
plt.title('Shell_weight по возрастным группам')
plt.xlabel('Возрастная группа')
plt.ylabel('Вес раковины')

plt.subplot(1, 3, 3)
age_group_sex = pd.crosstab(df['Age_Group'], df['Sex'], normalize='index') * 100
age_group_sex.plot(kind='bar', stacked=True)
plt.title('Распределение полов по возрастным группам')
plt.xlabel('Возрастная группа')
plt.ylabel('Процент')
plt.legend(title='Пол')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("\n8. КЛЮЧЕВЫЕ ВЫВОДЫ")
print("-" * 40)
print("✓ Набор данных не содержит пропущенных значений")
print("✓ Имеется 4177 записей с физическими параметрами абалонов")
print("✓ Целевая переменная 'Rings' имеет диапазон от 1 до 29")
print("✓ Обнаружены выбросы в признаке 'Height'")
print("✓ Наибольшая корреляция с целевой переменной у 'Shell_weight'")
print("✓ Данные подходят для задач регрессии и классификации")