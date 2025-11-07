# Задача 0. Введение в Python 3.

# 1) Подгрузить библиотеку Numpy
import numpy as np

# 2) Создать массив от 1 до 10 с шагом 2 тремя разными способами
arr1 = np.array([1, 3, 5, 7, 9])
arr2 = np.arange(1, 10, 2)
arr3 = np.linspace(1, 9, 5, dtype=int)

print("Массив тремя способами:")
print("np.array():", arr1)
print("np.arange():", arr2)
print("np.linspace():", arr3)

# 3) Создать массив целых чисел от 0 до 100 максимально короткой записью
k = np.arange(101)
print("\nМассив от 0 до 100:", k)

# 4) Создать массив x = sin(2π * (f0/fs) * k)
f0_fs = 0.07
x = np.sin(2 * np.pi * f0_fs * k)
print("\nМассив x[k]:", x[:10])  # Показать первые 10 элементов

# 5) Подгрузить модуль pyplot
import matplotlib.pyplot as plt

# 6) Создать холст размером 12 на 5 дюймов
plt.figure(figsize=[12, 5])

# 7) Построить график x[k]
plt.plot(k, x)
plt.xlabel("k")
plt.ylabel("x[k]")
plt.title("График x[k] = sin(2π * 0.07 * k)")
plt.grid(True)
plt.show()

# 8) Поменять цвет линии на красный, тип на пунктир и построить отсчеты
plt.figure(figsize=[12, 5])
plt.plot(k, x, '--r', label='x[k] (пунктир)')  # Красный пунктир
plt.stem(k, x, linefmt='b', markerfmt='bo', basefmt=' ', label='Отсчеты x[k]')
plt.xlabel("k")
plt.ylabel("x[k]")
plt.title("График x[k] с отсчетами")
plt.legend()
plt.grid(True)
plt.show()

# 9) Создать функцию sin(x)/x
def my_fun(x):
    if x == 0:
        return 1
    else:
        return np.sin(x) / x

# Создать массив значений для x от -20 до 20 с шагом 0.1
x_values = np.arange(-20, 20, 0.1)

# Применить функцию к массиву
y_values = np.array([my_fun(val) for val in x_values])

# Построить график функции sin(x)/x
plt.figure(figsize=[12, 5])
plt.plot(x_values, y_values, 'b-', linewidth=2)
plt.xlabel("x")
plt.ylabel("sin(x)/x")
plt.title("График функции sin(x)/x")
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)  # Горизонтальная линия y=0
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)  # Вертикальная линия x=0
plt.show()

# 10) Создать массив комплексной экспоненциальной функции
z = np.exp(-1j * 2 * np.pi * f0_fs * k)

# Построить графики реальной и мнимой частей
plt.figure(figsize=[12, 5])
plt.plot(k, z.real, 'g', label='Re(z[k])')
plt.plot(k, z.imag, 'r', label='Im(z[k])')
plt.xlabel("k")
plt.ylabel("z[k]")
plt.title("График реальной и мнимой частей z[k]")
plt.legend()
plt.grid(True)
plt.show()