# Задача 1.1. Дискретизация и квантование.
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.integrate as integrate

def quantize_uniform(x, quant_min=-1, quant_max=1, quant_level=5):
    """Uniform quantization approach
    From: Müller M. Fundamentals of music processing: Audio, analysis, algorithms, applications. – Springer, 2015.
    Notebook: C2S2_DigitalSignalQuantization.ipynb
    Args:
        x: Original signal
        quant_min: Minimum quantization level
        quant_max: Maximum quantization level
        quant_level: Number of quantization levels
    Returns:
        x_quant: Quantized signal
    """
    x_normalize = (x-quant_min) * (quant_level-1) / (quant_max-quant_min)
    x_normalize[x_normalize > quant_level - 1] = quant_level - 1
    x_normalize[x_normalize < 0] = 0
    x_normalize_quant = np.around(x_normalize)
    x_quant = (x_normalize_quant) * (quant_max-quant_min) / (quant_level-1) + quant_min
    return x_quant


# Параметры для варианта 7
N = 25                                    # Число отсчетов по времени
f0 = 1200.0                               # Частота синусоиды в Гц
fs = 6000.0                               # Частота дискретизации в Гц
k = np.arange(N)                          # Массив времен k от 0 до N-1 с шагом 1
x = np.sin(2 * np.pi * (f0 / fs) * k)     # Последовательность x[k]

# Параметры квантования
num_levels_list = [2, 4, 8, 16, 32]      # различные числа уровней квантования

print(f"Параметры сигнала:")
print(f"f0 = {f0} Гц")
print(f"fs = {fs} Гц")
print(f"f0/fs = {f0/fs:.4f}")
print(f"Период дискретизации T = {1/fs:.6f} с")
print(f"Длительность сигнала = {N/fs:.4f} с")

# 1) Анализ соотношения расстояния между отсчетами и частоты дискретизации
T = 1 / fs  # Период дискретизации
print(f"\n1) Соотношение расстояния между отсчетами и частоты дискретизации:")
print(f"Расстояние между отсчетами по времени: Δt = {T:.6f} с")
print(f"Частота дискретизации: fs = {fs} Гц")
print(f"Соотношение: Δt = 1/fs = {T:.6f} с")

# Построение графиков аналогового и дискретизованного сигнала
plt.figure(figsize=[12, 6])

# Аналоговый сигнал (более высокая частота дискретизации для плавного отображения)
t_analog = np.linspace(0, (N-1)/fs, num=1000)
x_analog = np.sin(2 * np.pi * f0 * t_analog)

plt.subplot(2, 1, 1)
plt.plot(t_analog * 1000, x_analog, 'g', linewidth=2, label='Аналоговый сигнал $x(t)$')  
plt.stem(k/fs * 1000, x, linefmt='b', markerfmt='bo', basefmt=' ', label='Дискретизованный сигнал $x[k]$')
plt.grid(True, alpha=0.3)
plt.xlabel("Время, мс")
plt.ylabel("Амплитуда")
plt.title("Аналоговый и дискретизованный сигналы")
plt.legend()

plt.subplot(2, 1, 2)
t_segment = np.linspace(0, 5/fs, num=500)  # Первые 5 отсчетов
x_segment = np.sin(2 * np.pi * f0 * t_segment)
plt.plot(t_segment * 1000, x_segment, 'g', linewidth=2, label='Аналоговый сигнал')
plt.stem(k[:6]/fs * 1000, x[:6], linefmt='b', markerfmt='bo', basefmt=' ', label='Отсчеты')
for i in range(6):
    plt.text(k[i]/fs * 1000, x[i] + 0.1, f'k={i}', ha='center', fontsize=8)
plt.grid(True, alpha=0.3)
plt.xlabel("Время, мс")
plt.ylabel("Амплитуда")
plt.title("Увеличенный фрагмент (первые 6 отсчетов)")
plt.legend()

plt.tight_layout()
plt.show()

# 2) Анализ ошибки квантования при различном числе уровней
print(f"\n2) Анализ ошибки квантования:")

plt.figure(figsize=[15, 10])

max_errors = []

for i, num_levels in enumerate(num_levels_list):
    # Квантование сигнала
    y = quantize_uniform(x, quant_min=-1, quant_max=1, quant_level=num_levels)
    
    # Вычисление ошибки квантования
    error = np.abs(x - y)
    max_error = np.max(error)
    max_errors.append(max_error)
    
    # Уровни квантования
    bins = np.linspace(-1, 1, num_levels + 1)
    quantization_levels = (bins[:-1] + bins[1:]) / 2
    
    print(f"Уровней квантования: {num_levels}")
    print(f"Максимальная ошибка: {max_error:.4f}")
    print(f"Уровни квантования: {quantization_levels}")
    print(f"Шаг квантования: {2/num_levels:.4f}")
    print("-" * 50)
    
    # Построение графиков
    plt.subplot(3, 2, i+1)
    
    # Исходный и квантованный сигнал
    plt.stem(k/fs * 1000, x, linefmt='b', markerfmt='bo', basefmt=' ', label='Исходный $x[k]$')
    plt.stem(k/fs * 1000, y, linefmt='r', markerfmt='ro', basefmt=' ', label='Квантованный $y[k]$')
    
    # Горизонтальные линии уровней квантования
    for level in quantization_levels:
        plt.axhline(y=level, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel("Время, мс")
    plt.ylabel("Амплитуда")
    plt.title(f"Квантование: {num_levels} уровней\nМакс. ошибка: {max_error:.4f}")
    plt.legend()

# График зависимости максимальной ошибки от числа уровней квантования
plt.subplot(3, 2, 6)
plt.plot(num_levels_list, max_errors, 'o-', linewidth=2, markersize=8)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Число уровней квантования (лог. шкала)")
plt.ylabel("Максимальная ошибка (лог. шкала)")
plt.title("Зависимость максимальной ошибки от числа уровней квантования")
plt.grid(True, alpha=0.3)

# Теоретическая зависимость (шаг квантования / 2)
theoretical_errors = [1/num_levels for num_levels in num_levels_list]
plt.plot(num_levels_list, theoretical_errors, 'r--', label='Теоретическая: 1/L')
plt.legend()

plt.tight_layout()
plt.show()

# Выводы
print(f"\nВЫВОДЫ:")
print(f"1) Расстояние между отсчетами Δt = {T:.6f} с обратно пропорционально")
print(f"   частоте дискретизации: Δt = 1/fs")
print(f"   Чем выше fs, тем меньше Δt и тем точнее представление сигнала")

print(f"\n2) Максимальная ошибка квантования уменьшается с ростом числа уровней:")
for num_levels, error in zip(num_levels_list, max_errors):
    print(f"   L = {num_levels}: ε_max = {error:.4f}")

print(f"\n3) Теоретически максимальная ошибка равномерного квантования:")
print(f"   ε_max = Δ/2, где Δ = 2/L - шаг квантования")
print(f"   Таким образом, ε_max = 1/L")

print(f"\n4) При увеличении числа уровней квантования в 2 раза,")
print(f"   максимальная ошибка уменьшается примерно в 2 раза")

# Дополнительный анализ - сравнение с теоремой Котельникова
f_max = fs / 2  # Теорема Котельникова: fs >= 2*f0
print(f"\nДополнительно: Проверка теоремы Котельникова")
print(f"f0 = {f0} Гц, fs = {fs} Гц, f_max по Котельникову = {f_max} Гц")
print(f"Условие fs >= 2*f0: {fs} >= {2*f0} -> {fs >= 2*f0}")
if fs >= 2*f0:
    print("Условие теоремы Котельникова выполняется")
else:
    print("Условие теоремы Котельникова НЕ выполняется!")