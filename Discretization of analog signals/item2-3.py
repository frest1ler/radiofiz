import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import windows

# Параметры для варианта 1
f0 = 10e3  # 10 кГц
tau = 500e-6  # 500 мкс

# Частота дискретизации (должна быть больше 2*f0 по теореме Найквиста)
fs = 10 * f0  # 100 кГц
N = int(fs * tau)  # Количество точек в сигнале
t = np.linspace(0, tau, N, endpoint=False)  # Временная ось

# 1. Сигнал с прямоугольным окном
x_rect = np.sin(2 * np.pi * f0 * t)

# 2. Сигнал с окном Ханна
hann_window = windows.hann(N)
x_hann = np.sin(2 * np.pi * f0 * t) * hann_window

# 3. Бесконечная синусоида (для сравнения, берем тот же отрезок времени)
x_periodic = np.sin(2 * np.pi * f0 * t)

# Вычисление преобразования Фурье
def compute_spectrum(x, fs):
    """Вычисляет спектр сигнала x"""
    N = len(x)
    # Применяем FFT с дополнением нулями для лучшего разрешения
    N_fft = 2**16
    X = fft(x, N_fft)
    # Частотная ось
    freqs = fftfreq(N_fft, 1/fs)
    # Возвращаем положительные частоты
    mask = freqs >= 0
    return freqs[mask], np.abs(X)[mask]

# Вычисляем спектры
freqs_rect, X_rect = compute_spectrum(x_rect, fs)
freqs_hann, X_hann = compute_spectrum(x_hann, fs)
freqs_per, X_per = compute_spectrum(x_periodic, fs)

# Построение графиков
plt.figure(figsize=(14, 10))

# Сигналы во временной области
plt.subplot(3, 2, 1)
plt.plot(t * 1e6, x_rect)
plt.title(f'Отрезок синусоиды (прямоугольное окно)\nf0={f0/1e3} кГц, τ={tau*1e6} мкс')
plt.xlabel('Время t, мкс')
plt.ylabel('x(t), В')
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(t * 1e6, x_hann)
plt.title('Отрезок синусоиды (окно Ханна)')
plt.xlabel('Время t, мкс')
plt.ylabel('x(t), В')
plt.grid(True)

# Спектры в линейном масштабе
plt.subplot(3, 2, 3)
plt.plot(freqs_rect / 1e3, X_rect * 1e6, label='Прямоугольное окно')
plt.axvline(x=f0/1e3, color='r', linestyle='--', alpha=0.5, label=f'f0={f0/1e3} кГц')
plt.title('Спектр (линейный масштаб)')
plt.xlabel('Частота f, кГц')
plt.ylabel('|X(f)|, мкВ/Гц')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(freqs_hann / 1e3, X_hann * 1e6, label='Окно Ханна')
plt.axvline(x=f0/1e3, color='r', linestyle='--', alpha=0.5, label=f'f0={f0/1e3} кГц')
plt.title('Спектр (линейный масштаб)')
plt.xlabel('Частота f, кГц')
plt.ylabel('|X(f)|, мкВ/Гц')
plt.legend()
plt.grid(True)

# Спектры в логарифмическом масштабе (для лучшего сравнения боковых лепестков)
plt.subplot(3, 2, 5)
plt.plot(freqs_rect / 1e3, 20 * np.log10(X_rect + 1e-12), label='Прямоугольное окно')
plt.plot(freqs_hann / 1e3, 20 * np.log10(X_hann + 1e-12), label='Окно Ханна', alpha=0.8)
plt.axvline(x=f0/1e3, color='r', linestyle='--', alpha=0.5, label=f'f0={f0/1e3} кГц')
plt.title('Спектр (логарифмический масштаб)')
plt.xlabel('Частота f, кГц')
plt.ylabel('|X(f)|, дБ')
plt.legend()
plt.grid(True)
plt.ylim([-100, 100])

# Сравнение с идеальным периодическим сигналом
plt.subplot(3, 2, 6)
plt.plot(freqs_per / 1e3, X_per * 1e6, label='Бесконечная синусоида')
plt.axvline(x=f0/1e3, color='r', linestyle='--', alpha=0.5, label=f'f0={f0/1e3} кГц')
plt.title('Спектр бесконечной синусоиды (теоретический)')
plt.xlabel('Частота f, кГц')
plt.ylabel('|X(f)|, мкВ/Гц')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Анализ результатов
print("=" * 60)
print("АНАЛИЗ РЕЗУЛЬТАТОВ")
print("=" * 60)

# 1. Положение главных максимумов
def find_peak(freqs, spectrum):
    """Находит положение максимума в спектре"""
    idx_max = np.argmax(spectrum)
    return freqs[idx_max]

peak_rect = find_peak(freqs_rect, X_rect)
peak_hann = find_peak(freqs_hann, X_hann)

print(f"1. Положение главных максимумов:")
print(f"   - Прямоугольное окно: {peak_rect/1e3:.2f} кГц (теоретическое: {f0/1e3} кГц)")
print(f"   - Окно Ханна: {peak_hann/1e3:.2f} кГц (теоретическое: {f0/1e3} кГц)")
print(f"   Отклонение от f0: {abs(peak_rect - f0)/f0*100:.2f}% для прямоугольного окна")
print()

# 2. Значение |X(f)| на частоте f0
def find_value_at_freq(freqs, spectrum, target_freq):
    """Находит значение спектра на заданной частоте"""
    idx = np.argmin(np.abs(freqs - target_freq))
    return spectrum[idx]

X_at_f0_rect = find_value_at_freq(freqs_rect, X_rect, f0)
X_at_f0_hann = find_value_at_freq(freqs_hann, X_hann, f0)

print(f"2. Значение |X(f)| на частоте f0={f0/1e3} кГц:")
print(f"   - Прямоугольное окно: {X_at_f0_rect*1e6:.2f} мкВ/Гц")
print(f"   - Окно Ханна: {X_at_f0_hann*1e6:.2f} мкВ/Гц")
print(f"   Отношение (Ханна/прямоугольное): {X_at_f0_hann/X_at_f0_rect:.3f}")
print()

# 3. Ширина главного лепестка на нулевом уровне
def find_main_lobe_width(freqs, spectrum, peak_freq, threshold=0.01):
    """Находит ширину главного лепестка на заданном уровне"""
    # Ищем частоты, где спектр опускается ниже threshold от максимума
    peak_idx = np.argmin(np.abs(freqs - peak_freq))
    peak_value = spectrum[peak_idx]
    
    # Ищем слева от пика
    left_idx = peak_idx
    while left_idx > 0 and spectrum[left_idx] > peak_value * threshold:
        left_idx -= 1
    
    # Ищем справа от пика
    right_idx = peak_idx
    while right_idx < len(spectrum) - 1 and spectrum[right_idx] > peak_value * threshold:
        right_idx += 1
    
    left_freq = freqs[left_idx] if left_idx > 0 else freqs[0]
    right_freq = freqs[right_idx] if right_idx < len(freqs) - 1 else freqs[-1]
    
    return right_freq - left_freq

width_rect = find_main_lobe_width(freqs_rect, X_rect, peak_rect, threshold=0.01)
width_hann = find_main_lobe_width(freqs_hann, X_hann, peak_hann, threshold=0.01)

print(f"3. Ширина главного лепестка на уровне -40 дБ (примерно 1% от максимума):")
print(f"   - Прямоугольное окно: {width_rect:.0f} Гц")
print(f"   - Окно Ханна: {width_hann:.0f} Гц")
print(f"   Отношение (Ханна/прямоугольное): {width_hann/width_rect:.2f}")
print()

# 4. Теоретические значения для сравнения
print("4. Теоретические значения:")
print(f"   а) Для прямоугольного окна:")
print(f"      - Ширина главного лепестка: 2/τ = {2/tau:.0f} Гц")
print(f"      - |X(f0)| ≈ τ/2 = {tau/2*1e6:.1f} мкВ/Гц")
print()
print(f"   б) Для окна Ханна:")
print(f"      - Ширина главного лепестка: 4/τ = {4/tau:.0f} Гц")
print(f"      - |X(f0)| ≈ 0.5 * (τ/2) = {0.5*tau/2*1e6:.1f} мкВ/Гц")
print()
print(f"   в) Для бесконечной синусоиды:")
print(f"      - Спектр представляет собой дельта-функции на ±f0")
print(f"      - Ширина главного лепестка: 0 Гц (идеально)")
print()

# 5. Сравнение боковых лепестков
def find_sidelobe_level(freqs, spectrum, peak_freq, main_lobe_width):
    """Находит уровень первого бокового лепестка"""
    # Исключаем главный лепесток
    mask = np.abs(freqs - peak_freq) > main_lobe_width/2
    side_freqs = freqs[mask]
    side_spectrum = spectrum[mask]
    
    # Ищем максимум в оставшейся области
    if len(side_spectrum) > 0:
        side_max = np.max(side_spectrum)
        peak_value = spectrum[np.argmin(np.abs(freqs - peak_freq))]
        return 20 * np.log10(side_max/peak_value)
    return None

sidelobe_rect = find_sidelobe_level(freqs_rect, X_rect, peak_rect, width_rect)
sidelobe_hann = find_sidelobe_level(freqs_hann, X_hann, peak_hann, width_hann)

print(f"5. Уровень первого бокового лепестка относительно главного:")
print(f"   - Прямоугольное окно: {sidelobe_rect:.1f} дБ")
print(f"   - Окно Ханна: {sidelobe_hann:.1f} дБ")
print(f"   Окно Ханна подавляет боковые лепестки лучше на {abs(sidelobe_rect) - abs(sidelobe_hann):.1f} дБ")
print("=" * 60)