import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Параметры для варианта 1
N = 3
tau = 300e-6
T_period = 600e-6
A = 1.0

# Общая длительность сигнала
total_duration = (N - 1) * T_period + tau

# Частота дискретизации
fs = 100 / tau
num_points = int(total_duration * fs)
num_points = max(num_points, 2**14)

# Временная ось
t = np.linspace(0, total_duration, num_points, endpoint=False)
dt = t[1] - t[0]

# Сигнал - пачка импульсов
signal = np.zeros(num_points)
for i in range(N):
    t_start = i * T_period
    t_end = t_start + tau
    idx = np.where((t >= t_start) & (t < t_end))[0]
    signal[idx] = A

# Спектр (БПФ)
fft_result = fft(signal) * dt
freqs = fftfreq(num_points, dt)

# Положительные частоты до 30 кГц
max_freq = 30e3
mask = (freqs >= 0) & (freqs <= max_freq)
freqs_pos = freqs[mask]
fft_pos = fft_result[mask]
spectrum_abs = np.abs(fft_pos)

# Аналитический спектр
def analytical_spectrum(f, N, tau, T_period, A=1.0):
    result = np.zeros_like(f)
    for i, freq in enumerate(f):
        if np.abs(freq) < 1e-12:
            result[i] = A * tau * N
        else:
            X_single = A * tau * np.abs(np.sin(np.pi * freq * tau) / (np.pi * freq * tau))
            if np.abs(np.sin(np.pi * freq * T_period)) < 1e-12:
                interference = N
            else:
                interference = np.abs(np.sin(np.pi * freq * N * T_period) / np.sin(np.pi * freq * T_period))
            result[i] = X_single * interference
    return result

analytical_spectrum_vals = analytical_spectrum(freqs_pos, N, tau, T_period, A)

# Построение только необходимых графиков
plt.figure(figsize=(14, 5))

# График 1: Сигнал
plt.subplot(1, 2, 1)
plt.plot(t * 1e6, signal, 'b-', linewidth=1.5)
plt.title(f'Пачка из N={N} прямоугольных импульсов\nτ={tau*1e6} мкс, T={T_period*1e6} мкс')
plt.xlabel('Время t, мкс')
plt.ylabel('x(t), В')
plt.grid(True, alpha=0.3)
plt.xlim([0, total_duration * 1e6])

# График 2: Спектры
plt.subplot(1, 2, 2)
plt.plot(freqs_pos / 1e3, spectrum_abs * 1e6, 'b-', linewidth=1.5, label='БПФ', alpha=0.8)
plt.plot(freqs_pos / 1e3, analytical_spectrum_vals * 1e6, 'r--', linewidth=2, label='Аналитический', alpha=0.8)
plt.title('Сравнение спектров')
plt.xlabel('Частота f, кГц')
plt.ylabel('|X(f)|, мкВ/Гц')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, max_freq / 1e3])

plt.tight_layout()
plt.show()

# Вывод информации
print("=" * 70)
print("РЕЗУЛЬТАТЫ ДЛЯ ВАРИАНТА 1")
print("=" * 70)
print(f"Параметры: N = {N}, τ = {tau*1e6} мкс, T_period = {T_period*1e6} мкс")
print()

# Основные характеристики
f0 = 1 / T_period
print(f"Основная частота повторения: f₀ = 1/T = {f0/1e3:.3f} кГц")
print(f"Расстояние между спектральными компонентами: Δf = {f0/1e3:.3f} кГц")
print(f"Ширина главного лепестка огибающей: 2/τ = {2/tau/1e3:.1f} кГц")
print()

# Амплитуды
print("Амплитуды на характерных частотах:")
print(f"  f = 0: |X(0)| = {spectrum_abs[0]*1e6:.1f} мкВ/Гц (теор.: {A*tau*N*1e6:.1f} мкВ/Гц)")

# Положения максимумов
print("\nПоложения первых максимумов:")
for k in range(1, 6):
    f_k = k * f0
    if f_k <= max_freq:
        idx = np.argmin(np.abs(freqs_pos - f_k))
        X_k = spectrum_abs[idx]
        print(f"  f = {f_k/1e3:.3f} кГц: |X(f)| = {X_k*1e6:.1f} мкВ/Гц")
print("=" * 70)