import numpy as np
import matplotlib.pyplot as plt

# Вариант 1 (0+1 = 1)
Nv = 0

# ============================================
# ЗАДАЧА 2.1 и 2.2: Последовательность x[k]
# ============================================
x = np.array([-0.17955181,  0.03363767, -0.18337288,  0.16530822, -0.47484624,
              0.38160159, -0.07694653,  0.17859373, -0.15940822,  0.27243374,
              0.20263674, -0.08910899,  0.41999793, -0.3445067 ,  0.1850142 ,
              -0.32245184, -0.11134278, -0.33976998,  0.30679677, -0.27035383,
              0.02273391, -0.41012909, -0.2271223 , -0.38985271, -0.36219159,
              0.49506967, -0.25982865, -0.08639243,  0.4594335 , -0.48944217,
              0.0304403 , -0.21047998])

N = len(x)

# ============================================
# ЗАДАЧА 2.1: Алгоритмы вычисления ДПФ
# ============================================
# а) Матричная форма ДПФ
def DFT_matrix(x):
    """Вычисление ДПФ с использованием матричной формы"""
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-2j * np.pi * k * n / N)
    return W @ x

# б) Алгоритм БПФ
def FFT(x):
    """Реализация алгоритма БПФ (рекурсивная, радикс-2)"""
    N = len(x)
    if N <= 1:
        return x
    even = FFT(x[0::2])
    odd = FFT(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N//2)]
    return [even[k] + T[k] for k in range(N//2)] + [even[k] - T[k] for k in range(N//2)]

# Вычисление ДПФ разными способами
X_matrix = DFT_matrix(x)
X_fft_custom = np.array(FFT(x))
X_fft_numpy = np.fft.fft(x)

# Сравнение результатов
print("="*60)
print("ЗАДАЧА 2.1: Сравнение алгоритмов вычисления ДПФ")
print("="*60)
print(f"Размер последовательности: N = {N}")
print(f"\nМаксимальная разница между матричным методом и БПФ (реализация): {np.max(np.abs(X_matrix - X_fft_custom)):.2e}")
print(f"Максимальная разница между матричным методом и numpy.fft: {np.max(np.abs(X_matrix - X_fft_numpy)):.2e}")
print(f"Максимальная разница между БПФ (реализация) и numpy.fft: {np.max(np.abs(X_fft_custom - X_fft_numpy)):.2e}")

# ============================================
# ЗАДАЧА 2.2: Свойства симметрии ДПФ
# ============================================
# Используем результат из numpy.fft для точности
X = X_fft_numpy

# Свойства симметрии для вещественного x[k]:
# 1. Re[X[n]] = Re[X[N-n]] (четная симметрия)
# 2. Im[X[n]] = -Im[X[N-n]] (нечетная симметрия)
# 3. |X[n]| = |X[N-n]| (четная симметрия)
# 4. arg(X[n]) = -arg(X[N-n]) (нечетная симметрия)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
n = np.arange(N)

# Действительная часть
axes[0, 0].stem(n, X.real, linefmt='C0-', markerfmt='C0o', basefmt='C0-')
axes[0, 0].set_xlabel('n')
axes[0, 0].set_ylabel('Re X[n]')
axes[0, 0].set_title('Задача 2.2: Действительная часть ДПФ')
axes[0, 0].grid(True, alpha=0.3)

# Мнимая часть
axes[0, 1].stem(n, X.imag, linefmt='C1-', markerfmt='C1o', basefmt='C1-')
axes[0, 1].set_xlabel('n')
axes[0, 1].set_ylabel('Im X[n]')
axes[0, 1].set_title('Задача 2.2: Мнимая часть ДПФ')
axes[0, 1].grid(True, alpha=0.3)

# Модуль
axes[1, 0].stem(n, np.abs(X), linefmt='C2-', markerfmt='C2o', basefmt='C2-')
axes[1, 0].set_xlabel('n')
axes[1, 0].set_ylabel('|X[n]|')
axes[1, 0].set_title('Задача 2.2: Модуль ДПФ')
axes[1, 0].grid(True, alpha=0.3)

# Фаза
axes[1, 1].stem(n, np.angle(X), linefmt='C3-', markerfmt='C3o', basefmt='C3-')
axes[1, 1].set_xlabel('n')
axes[1, 1].set_ylabel('∠ X[n] (рад)')
axes[1, 1].set_title('Задача 2.2: Фаза ДПФ')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Проверка свойств симметрии
print("\n" + "="*60)
print("ЗАДАЧА 2.2: Проверка свойств симметрии ДПФ")
print("="*60)

# Для вещественного сигнала должно выполняться:
# X[N-n] = conj(X[n]) для n=1..N-1
errors = []
for i in range(1, N):
    expected = np.conj(X[i])
    actual = X[N-i] if i != 0 else X[0]
    error = np.abs(expected - actual)
    errors.append(error)

max_error = np.max(errors)
print(f"Максимальная ошибка в свойстве симметрии X[N-n] = conj(X[n]): {max_error:.2e}")
print("Свойства симметрии выполняются с хорошей точностью" if max_error < 1e-10 else "Есть значительные отклонения от свойств симметрии")

# ============================================
# ЗАДАЧА 2.3: Циклический сдвиг в ДПФ
# ============================================
# Параметр m
m = max(Nv % 8, 1)
print(f"\n" + "="*60)
print(f"ЗАДАЧА 2.3: Циклический сдвиг (m = {m})")
print("="*60)

# Последовательность q[k]
N_q = 8
k = np.arange(N_q)
q = np.zeros(16)  # Нули при k >= 8
q[:N_q] = k + 1  # q[k] = k+1 при 0 <= k < 8

# Вычисление ДПФ Q[n]
Q = np.fft.fft(q[:N_q])  # ДПФ только первых 8 точек

# Вычисление Y[n] = exp(-j*2π/8 * m * n) * Q[n]
n = np.arange(N_q)
Y = Q * np.exp(-2j * np.pi * m * n / N_q)

# Обратное ДПФ для получения y[k]
y = np.fft.ifft(Y)

# Проверка: y должен быть циклически сдвинутой версией q
q_shifted_expected = np.roll(q[:N_q], m)  # Циклический сдвиг вправо на m

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Исходная последовательность q[k]
axes[0].stem(np.arange(16), q, linefmt='C0-', markerfmt='C0o', basefmt='C0-')
axes[0].axvline(x=7.5, color='red', linestyle='--', alpha=0.5, label='k=7')
axes[0].set_xlabel('k')
axes[0].set_ylabel('q[k]')
axes[0].set_title(f'Задача 2.3: Исходная последовательность q[k]')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Полученная последовательность y[k] (первые 8 точек)
axes[1].stem(np.arange(N_q), y.real, linefmt='C1-', markerfmt='C1o', basefmt='C1-', label='y[k] (результат)')
axes[1].stem(np.arange(N_q), q_shifted_expected, linefmt='C2--', markerfmt='C2x', basefmt='C2-', label='Ожидаемый сдвиг')
axes[1].set_xlabel('k')
axes[1].set_ylabel('y[k]')
axes[1].set_title(f'Задача 2.3: Результат y[k] (m={m})')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Модуль Y[n] (ДПФ сдвинутой последовательности)
axes[2].stem(np.arange(N_q), np.abs(Y), linefmt='C3-', markerfmt='C3o', basefmt='C3-')
axes[2].set_xlabel('n')
axes[2].set_ylabel('|Y[n]|')
axes[2].set_title(f'Задача 2.3: Модуль ДПФ Y[n]')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Проверка точности
print(f"\nСравнение полученного y[k] с ожидаемым циклическим сдвигом на m={m}:")
error = np.max(np.abs(y.real - q_shifted_expected))
print(f"Максимальная ошибка: {error:.2e}")
print("Сдвиг выполнен корректно" if error < 1e-10 else "Есть расхождения")

# Вывод теоретического объяснения
print("\n" + "="*60)
print("Теоретическое объяснение (Задача 2.3):")
print("="*60)
print("Теорема о циклическом сдвиге в ДПФ:")
print("Если y[k] = x[(k-m) mod N], то Y[n] = X[n] * exp(-j*2π*m*n/N)")
print(f"В данном случае: y[k] = q[(k-{m}) mod 8]")
print("Результат подтверждает теорему о циклическом сдвиге.")