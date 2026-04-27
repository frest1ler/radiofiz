import numpy as np
import matplotlib.pyplot as plt

# Параметры варианта 1
N = 32
m0 = 1
m1 = 0.25

# ============================================
# ЗАДАЧА 3.1: Интерполяция ДВПФ добавлением нулевых отсчетов
# ============================================

# Создание исходной последовательности x[k]
k = np.arange(N)
x = np.sin(2 * np.pi / N * m0 * k) + np.sin(2 * np.pi / N * (m0 + m1) * k)

# Частотная ось для ДВПФ (непрерывная)
nu = np.linspace(-0.5, 0.5, 1000)

# Функция для вычисления ДВПФ
def compute_DVTF(x, nu):
    """Вычисление ДВПФ последовательности x для частот nu"""
    N_len = len(x)
    X_nu = np.zeros(len(nu), dtype=complex)
    for i, nu_val in enumerate(nu):
        X_nu[i] = np.sum(x * np.exp(-2j * np.pi * nu_val * np.arange(N_len)))
    return X_nu

# Вычисление ДВПФ и ДПФ для исходного сигнала
X_nu = compute_DVTF(x, nu)
X_DFT = np.fft.fft(x)

# Частоты ДПФ (бины)
n = np.arange(N)
nu_DFT = n / N  # нормированные частоты

# Построение графиков для исходного сигнала
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График 1: модули ДВПФ и ДПФ (исходный сигнал)
axes[0].plot(nu, np.abs(X_nu), label='|X(ν)| (ДВПФ)', linewidth=2)
axes[0].stem(nu_DFT, np.abs(X_DFT), linefmt='C1-', markerfmt='C1o', 
            basefmt='C1-', label='|X[n]| (ДПФ)')
axes[0].set_xlabel('ν (нормированная частота)')
axes[0].set_ylabel('Амплитуда')
axes[0].set_title('Задача 3.1: Исходный сигнал (N=32)')
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[0].set_xlim([0, 0.5])  # показываем только положительные частоты

# Отметим частоты синусоид
f1 = m0 / N
f2 = (m0 + m1) / N
axes[0].axvline(x=f1, color='red', linestyle='--', alpha=0.5, label=f'ν={f1:.3f}')
axes[0].axvline(x=f2, color='green', linestyle='--', alpha=0.5, label=f'ν={f2:.3f}')
axes[0].legend()

# Zero-padding: увеличиваем размер в 4 раза, чтобы частоты попали на бины
N_zp = 128  # 32 * 4
x_zp = np.zeros(N_zp)
x_zp[:N] = x

# Вычисление ДВПФ и ДПФ для сигнала с zero-padding
X_nu_zp = compute_DVTF(x_zp, nu)
X_DFT_zp = np.fft.fft(x_zp)

# Частоты ДПФ для zero-padding
n_zp = np.arange(N_zp)
nu_DFT_zp = n_zp / N_zp

# График 2: модули ДВПФ и ДПФ (с zero-padding)
axes[1].plot(nu, np.abs(X_nu_zp), label='|X(ν)| (ДВПФ)', linewidth=2)
axes[1].stem(nu_DFT_zp, np.abs(X_DFT_zp), linefmt='C1-', markerfmt='C1o', 
            basefmt='C1-', label='|X[n]| (ДПФ)')
axes[1].set_xlabel('ν (нормированная частота)')
axes[1].set_ylabel('Амплитуда')
axes[1].set_title('Задача 3.1: Zero-padding (N=128)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()
axes[1].set_xlim([0, 0.5])

# Отметим частоты синусоид (теперь они попадают на бины ДПФ)
bin1 = int(m0 * N_zp / N)  # m0/N соответствует бину m0*4
bin2 = int((m0 + m1) * N_zp / N)  # (m0+m1)/N соответствует бину (m0+m1)*4
f1_zp = bin1 / N_zp
f2_zp = bin2 / N_zp
axes[1].axvline(x=f1_zp, color='red', linestyle='--', alpha=0.5, label=f'ν={f1_zp:.3f}')
axes[1].axvline(x=f2_zp, color='green', linestyle='--', alpha=0.5, label=f'ν={f2_zp:.3f}')
axes[1].legend()

plt.tight_layout()
plt.show()

# ============================================
# ЗАДАЧА 3.2: ДВПФ и ДПФ периодической последовательности
# ============================================

# Создание периодических последовательностей
k_periodic = np.arange(N)

# Случай 1: m = m0 = 1
x1 = np.cos(2 * np.pi / N * m0 * k_periodic) + np.sin(2 * np.pi / N * m0 * k_periodic)

# Случай 2: m = m0 + m1 = 1.25
x2 = np.cos(2 * np.pi / N * (m0 + m1) * k_periodic) + np.sin(2 * np.pi / N * (m0 + m1) * k_periodic)

# Вычисление ДПФ
X1_DFT = np.fft.fft(x1)
X2_DFT = np.fft.fft(x2)

# Вычисление ДВПФ для сравнения
X1_DVTF = compute_DVTF(x1, nu_DFT)  # вычисляем в точках ДПФ
X2_DVTF = compute_DVTF(x2, nu_DFT)

# Построение графиков
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Графики для m = m0 = 1
axes[0, 0].stem(nu_DFT, X1_DFT.real, linefmt='C0-', markerfmt='C0o', basefmt='C0-')
axes[0, 0].set_xlabel('ν (нормированная частота)')
axes[0, 0].set_ylabel('Re X[n]')
axes[0, 0].set_title('Задача 3.2: Действительная часть ДПФ (m=1)')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].stem(nu_DFT, X1_DFT.imag, linefmt='C1-', markerfmt='C1o', basefmt='C1-')
axes[0, 1].set_xlabel('ν (нормированная частота)')
axes[0, 1].set_ylabel('Im X[n]')
axes[0, 1].set_title('Задача 3.2: Мнимая часть ДПФ (m=1)')
axes[0, 1].grid(True, alpha=0.3)

# Графики для m = m0 + m1 = 1.25
axes[1, 0].stem(nu_DFT, X2_DFT.real, linefmt='C2-', markerfmt='C2o', basefmt='C2-')
axes[1, 0].set_xlabel('ν (нормированная частота)')
axes[1, 0].set_ylabel('Re X[n]')
axes[1, 0].set_title('Задача 3.2: Действительная часть ДПФ (m=1.25)')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].stem(nu_DFT, X2_DFT.imag, linefmt='C3-', markerfmt='C3o', basefmt='C3-')
axes[1, 1].set_xlabel('ν (нормированная частота)')
axes[1, 1].set_ylabel('Im X[n]')
axes[1, 1].set_title('Задача 3.2: Мнимая часть ДПФ (m=1.25)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Сравнение ДПФ и ДВПФ
print("="*60)
print("ЗАДАЧА 3.2: Сравнение ДПФ и ДВПФ")
print("="*60)

# Для m = 1 (целое) - периодический случай
print("\nСлучай m = 1 (целое число):")
print("ДПФ должно иметь только две ненулевые компоненты")
print(f"Ненулевые компоненты ДПФ: X[{m0}] и X[{N-m0}]")

# Для m = 1.25 (не целое) - непериодический случай
print("\nСлучай m = 1.25 (не целое):")
print("ДПФ имеет множество ненулевых компонентов из-за утечки спектра")

# Аналитическое выражение для ДПФ
print("\n" + "="*60)
print("Аналитическая запись ДПФ:")
print("="*60)
print("Для x[k] = cos(2π/N * m * k) + sin(2π/N * m * k)")
print("Используя формулу Эйлера:")
print("cos(θ) = (e^{jθ} + e^{-jθ})/2")
print("sin(θ) = (e^{jθ} - e^{-jθ})/(2j)")
print("\nТогда:")
print("x[k] = (1/2 - j/2) * exp(j*2π/N*m*k) + (1/2 + j/2) * exp(-j*2π/N*m*k)")
print("\nДПФ: X[n] = Σ_{k=0}^{N-1} x[k] * exp(-j*2π/N*n*k)")
print("Для целого m:")
print("X[n] = N/2 * [(1 - j) * δ[n - m] + (1 + j) * δ[n - (N-m)]]")
print("\nгде δ[n] - символ Кронекера (δ[n]=1 при n=0, иначе 0)")

# Проверка связи весов дельта-функций и величин отсчетов ДПФ
print("\n" + "="*60)
print("Проверка связи для m = 1:")
print("="*60)

# Находим индексы ненулевых компонент
idx1 = m0
idx2 = N - m0

print(f"X[{idx1}] = {X1_DFT[idx1]:.6f}")
print(f"X[{idx2}] = {X1_DFT[idx2]:.6f}")

# Ожидаемые значения по аналитической формуле
expected_X1 = N/2 * (1 - 1j)
expected_X2 = N/2 * (1 + 1j)

print(f"\nОжидаемое X[{idx1}] = N/2 * (1 - j) = {expected_X1:.6f}")
print(f"Ожидаемое X[{idx2}] = N/2 * (1 + j) = {expected_X2:.6f}")

# Проверка точности
error1 = np.abs(X1_DFT[idx1] - expected_X1)
error2 = np.abs(X1_DFT[idx2] - expected_X2)

print(f"\nПогрешность для X[{idx1}]: {error1:.2e}")
print(f"Погрешность для X[{idx2}]: {error2:.2e}")

# График для сравнения ДПФ и ДВПФ
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Модули ДПФ и ДВПФ для m = 1
axes[0].stem(nu_DFT, np.abs(X1_DFT), linefmt='C0-', markerfmt='C0o', 
            basefmt='C0-', label='|X[n]| (ДПФ)')
axes[0].plot(nu_DFT, np.abs(X1_DVTF), 'C1--', label='|X(ν)| (ДВПФ)', alpha=0.7)
axes[0].set_xlabel('ν (нормированная частота)')
axes[0].set_ylabel('Амплитуда')
axes[0].set_title('Задача 3.2: Сравнение ДПФ и ДВПФ (m=1)')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Модули ДПФ и ДВПФ для m = 1.25
axes[1].stem(nu_DFT, np.abs(X2_DFT), linefmt='C2-', markerfmt='C2o', 
            basefmt='C2-', label='|X[n]| (ДПФ)')
axes[1].plot(nu_DFT, np.abs(X2_DVTF), 'C3--', label='|X(ν)| (ДВПФ)', alpha=0.7)
axes[1].set_xlabel('ν (нормированная частота)')
axes[1].set_ylabel('Амплитуда')
axes[1].set_title('Задача 3.2: Сравнение ДПФ и ДВПФ (m=1.25)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Выводы:")
print("="*60)
print("1. Для целого m: ДПФ точно соответствует аналитической формуле")
print("   и состоит из двух ненулевых отсчетов.")
print("2. Для нецелого m: возникает утечка спектра, и ДПФ содержит")
print("   множество ненулевых компонент.")
print("3. Связь между ДПФ и ДВПФ: ДПФ представляет собой дискретные")
print("   отсчеты ДВПФ в точках ν = n/N.")