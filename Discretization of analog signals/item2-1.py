import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import integrate  

def quantize_uniform(x, quant_min=-1, quant_max=1, quant_level=5):
    """Uniform quantization approach"""
    x_normalize = (x-quant_min) * (quant_level-1) / (quant_max-quant_min)
    x_normalize[x_normalize > quant_level - 1] = quant_level - 1
    x_normalize[x_normalize < 0] = 0
    x_normalize_quant = np.around(x_normalize)
    x_quant = (x_normalize_quant) * (quant_max-quant_min) / (quant_level-1) + quant_min
    return x_quant

# Задача 2.1. Исследование прямоугольного импульса

print("Задача 2.1. Исследование прямоугольного импульса")
print("=" * 60)

# Параметры прямоугольного импульса
def rectangular_pulse(t, tau):
    """
    Прямоугольный импульс
    t - время
    tau - длительность импульса
    """
    return np.where((t >= -tau/2) & (t <= tau/2), 1, 0)

# Параметры для анализа
tau = 1.0  # длительность импульса
t1, t2 = -2, 2  # временной интервал для анализа

# Временная область
t = np.linspace(t1, t2, 1000)
x_t = rectangular_pulse(t, tau)

# Построение импульса во временной области
plt.figure(figsize=[15, 10])

plt.subplot(2, 2, 1)
plt.plot(t, x_t, 'b-', linewidth=2)
plt.xlabel("Время, с")
plt.ylabel("Амплитуда")
plt.title("Прямоугольный импульс во временной области")
plt.grid(True, alpha=0.3)
plt.axvline(x=-tau/2, color='r', linestyle='--', alpha=0.7, label=f'τ={tau} с')
plt.axvline(x=tau/2, color='r', linestyle='--', alpha=0.7)
plt.legend()

# Частотная область - аналитическое решение
f_analytic = np.linspace(-10, 10, 1000)
X_f_analytic = tau * np.sinc(f_analytic * tau)

plt.subplot(2, 2, 2)
plt.plot(f_analytic, np.abs(X_f_analytic), 'r-', linewidth=2, label='|X(f)|')
plt.plot(f_analytic, X_f_analytic.real, 'g--', alpha=0.7, label='Re(X(f))')
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда")
plt.title("Спектр прямоугольного импульса (аналитический)")
plt.grid(True, alpha=0.3)
plt.legend()

# Численное преобразование Фурье
def fourier_transform_numeric(signal_func, f_band, tau, t1, t2, num_points=1000):
    """
    Численное преобразование Фурье
    """
    X_f_numeric = []
    for f in f_band:
        integrand_real = lambda t: signal_func(t, tau) * np.cos(2 * np.pi * f * t)
        integrand_imag = lambda t: signal_func(t, tau) * np.sin(2 * np.pi * f * t)
        
        real_part = integrate.quad(integrand_real, t1, t2)[0]
        imag_part = integrate.quad(integrand_imag, t1, t2)[0]
        
        X_f_numeric.append(real_part + 1j * imag_part)
    
    return np.array(X_f_numeric)

# Вычисление численного преобразования Фурье
f_numeric = np.linspace(-10, 10, 200)
X_f_numeric = fourier_transform_numeric(rectangular_pulse, f_numeric, tau, t1, t2)

plt.subplot(2, 2, 3)
plt.plot(f_numeric, np.abs(X_f_numeric), 'ro-', markersize=3, label='Численное |X(f)|')
plt.plot(f_analytic, np.abs(X_f_analytic), 'b-', alpha=0.5, label='Аналитическое |X(f)|')
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда")
plt.title("Сравнение аналитического и численного спектра")
plt.grid(True, alpha=0.3)
plt.legend()

# Анализ различных длительностей импульса
tau_values = [0.5, 1.0, 2.0]
colors = ['red', 'blue', 'green']

plt.subplot(2, 2, 4)
for i, tau_val in enumerate(tau_values):
    X_f_tau = tau_val * np.sinc(f_analytic * tau_val)
    plt.plot(f_analytic, np.abs(X_f_tau), color=colors[i], 
             label=f'τ={tau_val} с', linewidth=2)

plt.xlabel("Частота, Гц")
plt.ylabel("|X(f)|")
plt.title("Спектр для различной длительности импульса")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# Анализ свойств спектра
print(f"\nАНАЛИЗ СВОЙСТВ СПЕКТРА ПРЯМОУГОЛЬНОГО ИМПУЛЬСА:")
print("-" * 50)

# Нули спектра
zeros_analytic = [n/tau for n in range(1, 6) if n/tau <= 10]
print(f"Нули спектра (f ≠ 0): {[f'{zero:.2f} Гц' for zero in zeros_analytic]}")

# Ширина главного лепестка
main_lobe_width = 2/tau
print(f"Ширина главного лепестка: {main_lobe_width:.2f} Гц")

# Максимум спектра
max_spectrum = tau
print(f"Максимум спектра (f=0): {max_spectrum:.2f}")

# Дополнительный анализ: энергия сигнала
print(f"\nЭНЕРГЕТИЧЕСКИЙ АНАЛИЗ:")
energy_time = integrate.quad(lambda t: rectangular_pulse(t, tau)**2, t1, t2)[0]
print(f"Энергия во временной области: {energy_time:.4f}")

# По теореме Парсеваля
energy_freq = integrate.quad(lambda f: (tau * np.sinc(f * tau))**2, -np.inf, np.inf)[0]
print(f"Энергия в частотной области (Парсеваль): {energy_freq:.4f}")

# Анализ дискретизации прямоугольного импульса
print(f"\nДИСКРЕТИЗАЦИЯ ПРЯМОУГОЛЬНОГО ИМПУЛЬСА:")
print("-" * 50)

# Параметры дискретизации
fs_values = [2/tau, 4/tau, 8/tau]  # различные частоты дискретизации
t_discrete = np.linspace(t1, t2, 1000)

plt.figure(figsize=[15, 5])

for i, fs in enumerate(fs_values):
    # Дискретизация
    T = 1/fs
    n = np.arange(int(t1/T), int(t2/T))
    t_n = n * T
    x_n = rectangular_pulse(t_n, tau)
    
    plt.subplot(1, 3, i+1)
    plt.plot(t_discrete, rectangular_pulse(t_discrete, tau), 'b-', 
             linewidth=2, label='Аналоговый', alpha=0.7)
    plt.stem(t_n, x_n, linefmt='r-', markerfmt='ro', basefmt=' ', 
             label=f'Дискретный (fs={fs:.1f} Гц)')
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда")
    plt.title(f"Дискретизация: fs={fs:.1f} Гц")
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.show()

# Анализ ошибки дискретизации
print(f"\nАНАЛИЗ ОШИБКИ ДИСКРЕТИЗАЦИИ:")
for fs in fs_values:
    T = 1/fs
    n = np.arange(int(t1/T), int(t2/T))
    t_n = n * T
    x_n = rectangular_pulse(t_n, tau)
    
    # Восстановление (идеальная интерполяция)
    t_recon = np.linspace(t1, t2, 1000)
    x_recon = np.zeros_like(t_recon)
    
    for t_point, x_point in zip(t_n, x_n):
        x_recon += x_point * np.sinc((t_recon - t_point) / T)
    
    error = np.max(np.abs(rectangular_pulse(t_recon, tau) - x_recon))
    print(f"fs = {fs:5.1f} Гц: максимальная ошибка = {error:.4f}")

# Анализ квантования прямоугольного импульса
print(f"\nКВАНТОВАНИЕ ПРЯМОУГОЛЬНОГО ИМПУЛЬСА:")
print("-" * 50)

# Дискретизированный импульс
fs = 10/tau  # высокая частота дискретизации
T = 1/fs
t_quant = np.linspace(-tau, tau, 50)
x_quant_original = rectangular_pulse(t_quant, tau)

quant_levels = [2, 4, 8, 16]

plt.figure(figsize=[15, 5])

for i, levels in enumerate(quant_levels):
    x_quantized = quantize_uniform(x_quant_original, quant_min=0, quant_max=1, 
                                  quant_level=levels)
    
    plt.subplot(1, 4, i+1)
    plt.stem(t_quant, x_quant_original, linefmt='b-', markerfmt='bo', 
             basefmt=' ', label='Исходный')
    plt.stem(t_quant, x_quantized, linefmt='r-', markerfmt='ro', 
             basefmt=' ', label=f'Квантованный (L={levels})')
    
    # Уровни квантования
    quant_levels_vals = np.linspace(0, 1, levels + 1)
    for level in quant_levels_vals:
        plt.axhline(y=level, color='gray', linestyle='--', alpha=0.3)
    
    error = np.max(np.abs(x_quant_original - x_quantized))
    plt.title(f'L={levels}, ошибка={error:.3f}')
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда")
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.show()

# Анализ ошибок квантования
print(f"\nАНАЛИЗ ОШИБОК КВАНТОВАНИЯ:")
for levels in quant_levels:
    x_quantized = quantize_uniform(x_quant_original, quant_min=0, quant_max=1, 
                                  quant_level=levels)
    max_error = np.max(np.abs(x_quant_original - x_quantized))
    rms_error = np.sqrt(np.mean((x_quant_original - x_quantized)**2))
    
    print(f"L = {levels:2d}: max ошибка = {max_error:.4f}, СКЗ = {rms_error:.4f}")

# Исследование спектральных искажений от квантования
print(f"\nСПЕКТРАЛЬНЫЕ ИСКАЖЕНИЯ ОТ КВАНТОВАНИЯ:")
print("-" * 50)

# Спектр исходного и квантованного сигнала
f_spectrum = np.linspace(-20, 20, 500)

# Исходный спектр
spectrum_original = tau * np.sinc(f_spectrum * tau)

# Спектр квантованного сигнала (для L=4)
x_quant_L4 = quantize_uniform(x_quant_original, quant_min=0, quant_max=1, quant_level=4)
# Аппроксимация спектра квантованного сигнала через FFT
spectrum_quantized = np.abs(np.fft.fftshift(np.fft.fft(x_quant_L4)))
f_quant = np.fft.fftshift(np.fft.fftfreq(len(x_quant_L4), T)) * fs

plt.figure(figsize=[12, 4])

plt.subplot(1, 2, 1)
plt.plot(f_spectrum, np.abs(spectrum_original), 'b-', linewidth=2, 
         label='Исходный спектр')
plt.plot(f_quant, spectrum_quantized/len(spectrum_quantized), 'r-', 
         label='Квантованный (L=4)')
plt.xlabel("Частота, Гц")
plt.ylabel("|X(f)|")
plt.title("Сравнение спектров")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
spectral_error = np.interp(f_quant, f_spectrum, np.abs(spectrum_original)) - spectrum_quantized/len(spectrum_quantized)
plt.plot(f_quant, spectral_error, 'g-', linewidth=2)
plt.xlabel("Частота, Гц")
plt.ylabel("Ошибка спектра")
plt.title("Спектральная ошибка от квантования")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ВЫВОДЫ
print(f"\nВЫВОДЫ ПО ЗАДАЧЕ 2.1:")
print("=" * 60)

print(f"\n1) СПЕКТР ПРЯМОУГОЛЬНОГО ИМПУЛЬСА:")
print(f"   - Имеет форму sinc-функции: X(f) = τ·sinc(fτ)")
print(f"   - Нули спектра: f = n/τ, n = ±1, ±2, ...")
print(f"   - Ширина главного лепестка: 2/τ Гц")
print(f"   - Чем короче импульс, тем шире спектр")

print(f"\n2) ДИСКРЕТИЗАЦИЯ:")
print(f"   - Для точного восстановления требуется fs > 2/τ")
print(f"   - При fs < 2/τ возникают значительные искажения")
print(f"   - Рекомендуемая fs ≥ 4/τ для качественного восстановления")

print(f"\n3) КВАНТОВАНИЕ:")
print(f"   - Прямоугольный импульс хорошо квантуется")
print(f"   - Максимальная ошибка ≈ шаг_квантования/2")
print(f"   - Для L=16 уровней ошибка < 3%")

print(f"\n4) СПЕКТРАЛЬНЫЕ ИСКАЖЕНИЯ:")
print(f"   - Квантование вносит высокочастотные составляющие")
print(f"   - Основная энергия сохраняется в главном лепестке")
print(f"   - Искажения наиболее заметны на краях спектра")

print(f"\n5) ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:")
print(f"   - Для импульсных сигналов: fs ≥ 4/τ_max")
print(f"   - Уровни квантования: L ≥ 16 для точности < 3%")
print(f"   - Учитывать ширину спектра при выборе fs")
