import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, fftfreq
from scipy.signal import find_peaks

# Параметры из условия (вариант 1: τ=100 мкс)
tau = 100e-6  # 100 мкс

# Функции окон во временной области
def rect_window(t, tau):
    return np.where(np.abs(t) <= tau/2, 1.0, 0.0)

def tri_window(t, tau):
    return np.where(np.abs(t) <= tau/2, 1 - np.abs(t)/(tau/2), 0.0)

def hann_window(t, tau):
    return np.where(np.abs(t) <= tau/2, 
                    0.5 * (1 + np.cos(2*np.pi * t / tau)), 
                    0.0)

# Аналитические выражения для спектральных плотностей
def rect_spectrum(f, tau):
    """Спектр прямоугольного окна: X(f) = τ * sinc(πfτ)"""
    # Избегаем деления на 0
    mask = np.abs(f) < 1e-10
    result = np.zeros_like(f, dtype=complex)
    result[mask] = tau  # предел при f->0
    result[~mask] = tau * np.sin(np.pi*f[~mask]*tau) / (np.pi*f[~mask]*tau)
    return result

def tri_spectrum(f, tau):
    """Спектр треугольного окна: X(f) = (τ/2) * sinc²(πfτ/2)"""
    # Избегаем деления на 0
    mask = np.abs(f) < 1e-10
    result = np.zeros_like(f, dtype=complex)
    result[mask] = tau/2  # предел при f->0
    result[~mask] = (tau/2) * (np.sin(np.pi*f[~mask]*tau/2) / (np.pi*f[~mask]*tau/2))**2
    return result

def hann_spectrum(f, tau):
    """Спектр окна Ханна: X(f) = 0.5*τ*sinc(πfτ) / (1 - (fτ)²)"""
    # Избегаем деления на 0
    mask = np.abs(f) < 1e-10
    result = np.zeros_like(f, dtype=complex)
    result[mask] = tau/2  # предел при f->0
    
    f_nonzero = f[~mask]
    tau_f = tau * f_nonzero
    result[~mask] = (tau/2) * np.sin(np.pi*tau_f) / (np.pi*tau_f) / (1 - tau_f**2)
    return result

# Диапазон частот для анализа
N = 8192
t = np.linspace(-tau, tau, N)
f = fftshift(fftfreq(N, t[1]-t[0]))

# Вычисляем спектры
rect_spec = rect_spectrum(f, tau)
tri_spec = tri_spectrum(f, tau)
hann_spec = hann_spectrum(f, tau)

# Модули спектров
rect_abs = np.abs(rect_spec)
tri_abs = np.abs(tri_spec)
hann_abs = np.abs(hann_spec)

# Нормализация относительно X(0)
rect_norm = rect_abs / rect_abs[f == 0]
tri_norm = tri_abs / tri_abs[f == 0]
hann_norm = hann_abs / hann_abs[f == 0]

# Поиск боковых лепестков
def find_first_sidelobe(spec, f_vals, main_lobe_width=1.5/tau):
    """Находит первый боковой лепесток и его уровень в дБ"""
    # Ищем только в области за пределами главного лепестка
    mask = np.abs(f_vals) > main_lobe_width
    spec_side = spec[mask]
    f_side = f_vals[mask]
    
    # Ищем пики (максимумы)
    peaks, properties = find_peaks(spec_side, height=0.01)
    
    if len(peaks) > 0:
        # Первый пик - это первый боковой лепесток
        first_peak_idx = peaks[0]
        peak_val = spec_side[first_peak_idx]
        # Уровень в дБ
        level_db = 20 * np.log10(peak_val)
        return level_db, f_side[first_peak_idx]
    return None, None

# Уровни первого бокового лепестка
rect_level, rect_f_sidelobe = find_first_sidelobe(rect_norm, f, main_lobe_width=1.2/tau)
tri_level, tri_f_sidelobe = find_first_sidelobe(tri_norm, f, main_lobe_width=2.2/tau)
hann_level, hann_f_sidelobe = find_first_sidelobe(hann_norm, f, main_lobe_width=2.2/tau)

# Ширина главного лепестка (по первым нулям)
def find_main_lobe_width(spec, f_vals, threshold=0.01):
    """Находит ширину главного лепестка по первым нулям"""
    # Ищем только положительные частоты
    pos_mask = f_vals >= 0
    spec_pos = spec[pos_mask]
    f_pos = f_vals[pos_mask]
    
    # Ищем точку, где спектр опускается ниже threshold
    # и затем снова поднимается (это будет за пределами главного лепестка)
    below_threshold = np.where(spec_pos < threshold)[0]
    if len(below_threshold) > 0:
        first_zero = below_threshold[0]
        # Первый минимум - это граница главного лепестка
        return 2 * f_pos[first_zero]  # умножаем на 2, т.к. симметрично
    return None

rect_width = find_main_lobe_width(rect_abs, f, threshold=rect_abs.max()*0.001)
tri_width = find_main_lobe_width(tri_abs, f, threshold=tri_abs.max()*0.001)
hann_width = find_main_lobe_width(hann_abs, f, threshold=hann_abs.max()*0.001)

# Вывод результатов в таблицу
print("Результаты для τ =", tau*1e6, "мкс")
print("="*70)
print(f"{'Окна':<15} {'Уровень 1-го бок. лепестка, дБ':<35} {'Ширина главн. лепестка, Гц':<30}")
print("="*70)
print(f"{'прямоугольное':<15} {rect_level:>8.2f}{'':<27} {rect_width:>8.1f}{'':<22}")
print(f"{'треугольное':<15} {tri_level:>8.2f}{'':<27} {tri_width:>8.1f}{'':<22}")
print(f"{'Ханна':<15} {hann_level:>8.2f}{'':<27} {hann_width:>8.1f}{'':<22}")
print("="*70)

# Построение графиков
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Временные области
axes[0, 0].plot(t*1e6, rect_window(t, tau))
axes[0, 0].set_title('Прямоугольное окно')
axes[0, 0].set_xlabel('Время t, мкс')
axes[0, 0].set_ylabel('w(t)')
axes[0, 0].grid(True)

axes[0, 1].plot(t*1e6, tri_window(t, tau))
axes[0, 1].set_title('Треугольное окно')
axes[0, 1].set_xlabel('Время t, мкс')
axes[0, 1].set_ylabel('w(t)')
axes[0, 1].grid(True)

axes[0, 2].plot(t*1e6, hann_window(t, tau))
axes[0, 2].set_title('Окно Ханна')
axes[0, 2].set_xlabel('Время t, мкс')
axes[0, 2].set_ylabel('w(t)')
axes[0, 2].grid(True)

# Частотные области (нормированные)
f_khz = f / 1e3  # переводим в кГц

axes[1, 0].plot(f_khz, 20*np.log10(rect_norm + 1e-10))
axes[1, 0].set_title('Спектр прямоугольного окна')
axes[1, 0].set_xlabel('Частота f, кГц')
axes[1, 0].set_ylabel('|X(f)|/X(0), дБ')
axes[1, 0].grid(True)
axes[1, 0].set_ylim([-100, 5])

axes[1, 1].plot(f_khz, 20*np.log10(tri_norm + 1e-10))
axes[1, 1].set_title('Спектр треугольного окна')
axes[1, 1].set_xlabel('Частота f, кГц')
axes[1, 1].set_ylabel('|X(f)|/X(0), дБ')
axes[1, 1].grid(True)
axes[1, 1].set_ylim([-100, 5])

axes[1, 2].plot(f_khz, 20*np.log10(hann_norm + 1e-10))
axes[1, 2].set_title('Спектр окна Ханна')
axes[1, 2].set_xlabel('Частота f, кГц')
axes[1, 2].set_ylabel('|X(f)|/X(0), дБ')
axes[1, 2].grid(True)
axes[1, 2].set_ylim([-100, 5])

plt.tight_layout()
plt.show()

# Аналитические формулы для теоретических нулей
print("\nТеоретические положения нулей:")
print("Прямоугольное окно: f = k/τ, k = ±1, ±2, ...")
print("Треугольное окно: f = 2k/τ, k = ±1, ±2, ...")
print("Окно Ханна: f = 2k/τ, k = ±1, ±2, ... (k ≠ 0)")

# Практические нули (из графика)
print("\nПрактические положения первых нулей (из графика):")
rect_zero_idx = np.where(np.abs(rect_norm[1:]) < 0.01)[0]
if len(rect_zero_idx) > 0:
    print(f"Прямоугольное окно: f ≈ {np.abs(f[rect_zero_idx[0]+1])/1e3:.2f} кГц")

tri_zero_idx = np.where(np.abs(tri_norm[1:]) < 0.01)[0]
if len(tri_zero_idx) > 0:
    print(f"Треугольное окно: f ≈ {np.abs(f[tri_zero_idx[0]+1])/1e3:.2f} кГц")

hann_zero_idx = np.where(np.abs(hann_norm[1:]) < 0.01)[0]
if len(hann_zero_idx) > 0:
    print(f"Окно Ханна: f ≈ {np.abs(f[hann_zero_idx[0]+1])/1e3:.2f} кГц")