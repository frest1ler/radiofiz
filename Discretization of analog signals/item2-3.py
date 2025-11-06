import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def quantize_uniform(x, quant_min=-1, quant_max=1, quant_level=5):
    """Uniform quantization approach"""
    x_normalize = (x-quant_min) * (quant_level-1) / (quant_max-quant_min)
    x_normalize[x_normalize > quant_level - 1] = quant_level - 1
    x_normalize[x_normalize < 0] = 0
    x_normalize_quant = np.around(x_normalize)
    x_quant = (x_normalize_quant) * (quant_max-quant_min) / (quant_level-1) + quant_min
    return x_quant

# Задача 2.3. Исследование гауссова импульса

print("Задача 2.3. Исследование гауссова импульса")
print("=" * 60)

# Параметры гауссова импульса
def gaussian_pulse(t, tau):
    """
    Гауссов импульс
    t - время
    tau - параметр, связанный с шириной импульса
    """
    return np.exp(-(t**2) / (2 * (tau/4)**2))

# Параметры для анализа
tau = 1.0  # параметр ширины импульса
t1, t2 = -3, 3  # временной интервал для анализа

# Временная область
t = np.linspace(t1, t2, 1000)
x_t = gaussian_pulse(t, tau)

# Построение импульса во временной области
plt.figure(figsize=[15, 10])

plt.subplot(2, 2, 1)
plt.plot(t, x_t, 'b-', linewidth=2)
plt.xlabel("Время, с")
plt.ylabel("Амплитуда")
plt.title("Гауссов импульс во временной области")
plt.grid(True, alpha=0.3)

# Ширина на уровне 1/e и 1/2
level_1e = 1/np.e
level_half = 0.5
t_1e = tau/4 * np.sqrt(2 * np.log(1/level_1e))
t_half = tau/4 * np.sqrt(2 * np.log(1/level_half))

plt.axhline(y=level_1e, color='r', linestyle='--', alpha=0.7, label=f'1/e ≈ {level_1e:.3f}')
plt.axhline(y=level_half, color='g', linestyle='--', alpha=0.7, label=f'1/2 = {level_half:.1f}')
plt.axvline(x=-t_1e, color='r', linestyle=':', alpha=0.5)
plt.axvline(x=t_1e, color='r', linestyle=':', alpha=0.5)
plt.axvline(x=-t_half, color='g', linestyle=':', alpha=0.5)
plt.axvline(x=t_half, color='g', linestyle=':', alpha=0.5)
plt.legend()

# Частотная область - аналитическое решение
f_analytic = np.linspace(-10, 10, 1000)
# Спектр гауссова импульса: также гауссова функция
sigma_t = tau/4  # стандартное отклонение во временной области
sigma_f = 1/(2 * np.pi * sigma_t)  # стандартное отклонение в частотной области
X_f_analytic = np.sqrt(2 * np.pi) * sigma_t * np.exp(-2 * (np.pi * f_analytic * sigma_t)**2)

plt.subplot(2, 2, 2)
plt.plot(f_analytic, np.abs(X_f_analytic), 'r-', linewidth=2, label='|X(f)|')
plt.plot(f_analytic, X_f_analytic.real, 'g--', alpha=0.7, label='Re(X(f))')
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда")
plt.title("Спектр гауссова импульса (аналитический)")
plt.grid(True, alpha=0.3)
plt.legend()

# Численное преобразование Фурье
def fourier_transform_numeric(signal_func, f_band, tau, t1, t2):
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
X_f_numeric = fourier_transform_numeric(gaussian_pulse, f_numeric, tau, t1, t2)

plt.subplot(2, 2, 3)
plt.plot(f_numeric, np.abs(X_f_numeric), 'ro-', markersize=3, label='Численное |X(f)|')
plt.plot(f_analytic, np.abs(X_f_analytic), 'b-', alpha=0.5, label='Аналитическое |X(f)|')
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда")
plt.title("Сравнение аналитического и численного спектра")
plt.grid(True, alpha=0.3)
plt.legend()

# Анализ различных параметров ширины импульса
tau_values = [0.5, 1.0, 2.0]
colors = ['red', 'blue', 'green']

plt.subplot(2, 2, 4)
for i, tau_val in enumerate(tau_values):
    sigma_t_val = tau_val/4
    X_f_tau = np.sqrt(2 * np.pi) * sigma_t_val * np.exp(-2 * (np.pi * f_analytic * sigma_t_val)**2)
    plt.plot(f_analytic, np.abs(X_f_tau), color=colors[i], 
             label=f'τ={tau_val} с', linewidth=2)

plt.xlabel("Частота, Гц")
plt.ylabel("|X(f)|")
plt.title("Спектр для различной ширины импульса")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# Анализ свойств спектра
print(f"\nАНАЛИЗ СВОЙСТВ СПЕКТРА ГАУССОВА ИМПУЛЬСА:")
print("-" * 50)

# Ширина на уровне 1/e во временной области
sigma_t = tau/4
width_time_1e = 2 * sigma_t * np.sqrt(2 * np.log(1/level_1e))
print(f"Ширина импульса на уровне 1/e: {width_time_1e:.2f} с")

# Ширина на уровне 1/e в частотной области
sigma_f = 1/(2 * np.pi * sigma_t)
width_freq_1e = 2 * sigma_f * np.sqrt(2 * np.log(1/level_1e))
print(f"Ширина спектра на уровне 1/e: {width_freq_1e:.2f} Гц")

# Произведение время-частота (принцип неопределенности)
time_bandwidth_product = width_time_1e * width_freq_1e
print(f"Произведение время-частота: {time_bandwidth_product:.4f}")

# Теоретический минимум по принципу неопределенности
uncertainty_minimum = 1/(4 * np.pi)
print(f"Теоретический минимум (1/4π): {uncertainty_minimum:.4f}")

# Сравнение с прямоугольным и треугольным импульсами
print(f"\nСРАВНЕНИЕ С ДРУГИМИ ИМПУЛЬСАМИ:")
rectangular_spectrum = tau * np.sinc(f_analytic * tau)
triangular_spectrum = (tau/2) * (np.sinc(f_analytic * tau/2))**2

plt.figure(figsize=[12, 5])

plt.subplot(1, 2, 1)
plt.plot(f_analytic, np.abs(X_f_analytic), 'b-', linewidth=2, label='Гауссов')
plt.plot(f_analytic, np.abs(rectangular_spectrum), 'r--', linewidth=2, label='Прямоугольный')
plt.plot(f_analytic, np.abs(triangular_spectrum), 'g-.', linewidth=2, label='Треугольный')
plt.xlabel("Частота, Гц")
plt.ylabel("|X(f)|")
plt.title("Сравнение спектров импульсов")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
# В логарифмическом масштабе
plt.semilogy(f_analytic, np.abs(X_f_analytic), 'b-', linewidth=2, label='Гауссов')
plt.semilogy(f_analytic, np.abs(rectangular_spectrum), 'r--', linewidth=2, label='Прямоугольный')
plt.semilogy(f_analytic, np.abs(triangular_spectrum), 'g-.', linewidth=2, label='Треугольный')
plt.xlabel("Частота, Гц")
plt.ylabel("|X(f)| (лог. шкала)")
plt.title("Сравнение спектров (лог. масштаб)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# Анализ скорости затухания спектра
print(f"\nАНАЛИЗ СКОРОСТИ ЗАТУХАНИЯ СПЕКТРА:")
f_large = np.linspace(5, 50, 1000)
spectrum_large_f = np.abs(np.sqrt(2 * np.pi) * sigma_t * np.exp(-2 * (np.pi * f_large * sigma_t)**2))

plt.figure(figsize=[10, 4])
plt.loglog(f_large, spectrum_large_f, 'b-', linewidth=2, label='Гауссов спектр')
plt.loglog(f_large, 1/(f_large**1), 'r--', linewidth=1, label='~1/f (прямоугольный)')
plt.loglog(f_large, 1/(f_large**2), 'g--', linewidth=1, label='~1/f² (треугольный)')
plt.loglog(f_large, np.exp(-f_large**2), 'm--', linewidth=1, label='~exp(-f²) (гауссов)')
plt.xlabel("Частота, Гц (лог. шкала)")
plt.ylabel("|X(f)| (лог. шкала)")
plt.title("Затухание спектра на высоких частотах")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print(f"Скорость затухания: экспоненциальная ~exp(-f²) (самая быстрая)")

# Энергетический анализ
print(f"\nЭНЕРГЕТИЧЕСКИЙ АНАЛИЗ:")
energy_time = integrate.quad(lambda t: gaussian_pulse(t, tau)**2, -np.inf, np.inf)[0]
print(f"Полная энергия во временной области: {energy_time:.4f}")

# Теоретическая энергия гауссова импульса
energy_theoretical = np.sqrt(np.pi) * sigma_t
print(f"Теоретическая энергия: {energy_theoretical:.4f}")

# По теореме Парсеваля
energy_freq = integrate.quad(lambda f: np.abs(np.sqrt(2 * np.pi) * sigma_t * 
                             np.exp(-2 * (np.pi * f * sigma_t)**2))**2, -np.inf, np.inf)[0]
print(f"Энергия в частотной области (Парсеваль): {energy_freq:.4f}")

# Дискретизация гауссова импульса
print(f"\nДИСКРЕТИЗАЦИЯ ГАУССОВА ИМПУЛЬСА:")
print("-" * 50)

# Определим эффективную ширину импульса (на уровне 0.01 от максимума)
effective_width = 2 * sigma_t * np.sqrt(2 * np.log(100))
fs_values = [2/effective_width, 4/effective_width, 8/effective_width, 16/effective_width]

plt.figure(figsize=[15, 10])

for i, fs in enumerate(fs_values):
    # Дискретизация
    T = 1/fs
    n = np.arange(int(t1/T), int(t2/T))
    t_n = n * T
    x_n = gaussian_pulse(t_n, tau)
    
    plt.subplot(2, 2, i+1)
    plt.plot(t, gaussian_pulse(t, tau), 'b-', linewidth=2, label='Аналоговый', alpha=0.7)
    plt.stem(t_n, x_n, linefmt='r-', markerfmt='ro', basefmt=' ', 
             label=f'Дискретный (fs={fs:.1f} Гц)')
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда")
    plt.title(f"Дискретизация: fs={fs:.1f} Гц")
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.show()

# Анализ ошибки дискретизации и восстановления
print(f"\nАНАЛИЗ ОШИБКИ ДИСКРЕТИЗАЦИИ И ВОССТАНОВЛЕНИЯ:")
for fs in fs_values:
    T = 1/fs
    n = np.arange(int(t1/T), int(t2/T))
    t_n = n * T
    x_n = gaussian_pulse(t_n, tau)
    
    # Восстановление (идеальная интерполяция)
    t_recon = np.linspace(t1, t2, 1000)
    x_recon = np.zeros_like(t_recon)
    
    for t_point, x_point in zip(t_n, x_n):
        x_recon += x_point * np.sinc((t_recon - t_point) / T)
    
    error = np.max(np.abs(gaussian_pulse(t_recon, tau) - x_recon))
    rms_error = np.sqrt(np.mean((gaussian_pulse(t_recon, tau) - x_recon)**2))
    
    print(f"fs = {fs:5.1f} Гц: max ошибка = {error:.6f}, СКЗ = {rms_error:.6f}")

# Квантование гауссова импульса
print(f"\nКВАНТОВАНИЕ ГАУССОВА ИМПУЛЬСА:")
print("-" * 50)

# Дискретизированный импульс
fs = 20/effective_width  # высокая частота дискретизации
T = 1/fs
t_quant = np.linspace(-2, 2, 200)
x_quant_original = gaussian_pulse(t_quant, tau)

quant_levels = [8, 16, 32, 64]

plt.figure(figsize=[15, 10])

for i, levels in enumerate(quant_levels):
    x_quantized = quantize_uniform(x_quant_original, quant_min=0, quant_max=1, 
                                  quant_level=levels)
    
    plt.subplot(2, 2, i+1)
    plt.plot(t_quant, x_quant_original, 'b-', linewidth=2, label='Исходный')
    plt.stem(t_quant, x_quantized, linefmt='r-', markerfmt='ro', basefmt=' ', 
             label=f'Квантованный (L={levels})')
    
    # Уровни квантования
    quant_levels_vals = np.linspace(0, 1, levels + 1)
    for level in quant_levels_vals[1:-1]:  # Пропускаем крайние уровни
        plt.axhline(y=level, color='gray', linestyle='--', alpha=0.3)
    
    error = np.max(np.abs(x_quant_original - x_quantized))
    plt.title(f'L={levels}, max ошибка={error:.4f}')
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда")
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.show()

# Анализ ошибок квантования
print(f"\nАНАЛИЗ ОШИБОК КВАНТОВАНИЯ:")
max_errors = []
rms_errors = []
snr_values = []

for levels in quant_levels:
    x_quantized = quantize_uniform(x_quant_original, quant_min=0, quant_max=1, 
                                  quant_level=levels)
    max_error = np.max(np.abs(x_quant_original - x_quantized))
    rms_error = np.sqrt(np.mean((x_quant_original - x_quantized)**2))
    
    # SNR calculation
    signal_power = np.mean(x_quant_original**2)
    noise_power = np.mean((x_quant_original - x_quantized)**2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    max_errors.append(max_error)
    rms_errors.append(rms_error)
    snr_values.append(snr)
    
    print(f"L = {levels:2d}: max ошибка = {max_error:.6f}, СКЗ = {rms_error:.6f}, SNR = {snr:6.2f} дБ")

# График зависимости ошибки от числа уровней
bits_per_sample = [np.log2(L) for L in quant_levels]
plt.figure(figsize=[12, 4])

plt.subplot(1, 2, 1)
plt.plot(quant_levels, max_errors, 'ro-', label='Макс. ошибка')
plt.plot(quant_levels, rms_errors, 'bs-', label='СКЗ ошибка')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Число уровней квантования")
plt.ylabel("Ошибка")
plt.title("Зависимость ошибки от числа уровней")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(bits_per_sample, snr_values, 'go-', linewidth=2)
plt.xlabel("Бит на отсчет")
plt.ylabel("SNR, дБ")
plt.title("Зависимость SNR от битрейта")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Спектральные искажения от квантования
print(f"\nСПЕКТРАЛЬНЫЕ ИСКАЖЕНИЯ ОТ КВАНТОВАНИЯ:")
print("-" * 50)

# Спектр исходного и квантованного сигнала
f_spectrum = np.linspace(-15, 15, 500)

# Исходный спектр
spectrum_original = np.sqrt(2 * np.pi) * sigma_t * np.exp(-2 * (np.pi * f_spectrum * sigma_t)**2)

# Спектр квантованного сигнала (для L=16)
x_quant_L16 = quantize_uniform(x_quant_original, quant_min=0, quant_max=1, quant_level=16)
spectrum_quantized = np.abs(np.fft.fftshift(np.fft.fft(x_quant_L16)))
f_quant = np.fft.fftshift(np.fft.fftfreq(len(x_quant_L16), T)) * fs

plt.figure(figsize=[12, 4])

plt.subplot(1, 2, 1)
plt.plot(f_spectrum, np.abs(spectrum_original), 'b-', linewidth=2, 
         label='Исходный спектр')
plt.plot(f_quant, spectrum_quantized/len(spectrum_quantized), 'r-', 
         label='Квантованный (L=16)')
plt.xlabel("Частота, Гц")
plt.ylabel("|X(f)|")
plt.title("Сравнение спектров")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
# Спектральная плотность ошибки
error_signal = x_quant_original - x_quant_L16
spectrum_error = np.abs(np.fft.fftshift(np.fft.fft(error_signal)))
plt.semilogy(f_quant, spectrum_error/len(spectrum_error), 'g-', linewidth=2)
plt.xlabel("Частота, Гц")
plt.ylabel("|E(f)| (лог. шкала)")
plt.title("Спектр ошибки квантования")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Принцип неопределенности для различных импульсов
print(f"\nПРИНЦИП НЕОПРЕДЕЛЕННОСТИ ДЛЯ РАЗЛИЧНЫХ ИМПУЛЬСОВ:")
print("-" * 50)

# Для прямоугольного импульса
rect_width_time = tau
rect_width_freq = 2/tau  # ширина главного лепестка
rect_uncertainty = rect_width_time * rect_width_freq
print(f"Прямоугольный: Δt·Δf = {rect_uncertainty:.4f}")

# Для треугольного импульса
triang_width_time = tau
triang_width_freq = 4/tau  # ширина главного лепестка
triang_uncertainty = triang_width_time * triang_width_freq
print(f"Треугольный: Δt·Δf = {triang_uncertainty:.4f}")

# Для гауссова импульса
gauss_uncertainty = time_bandwidth_product
print(f"Гауссов: Δt·Δf = {gauss_uncertainty:.4f}")

print(f"Теоретический минимум: 1/4π = {1/(4*np.pi):.4f}")

# ВЫВОДЫ
print(f"\nВЫВОДЫ ПО ЗАДАЧЕ 2.3:")
print("=" * 60)

print(f"\n1) СПЕКТР ГАУССОВА ИМПУЛЬСА:")
print(f"   - Также имеет гауссову форму")
print(f"   - Не имеет боковых лепестков")
print(f"   - Затухает экспоненциально ~exp(-f²)")

print(f"\n3) ДИСКРЕТИЗАЦИЯ:")
print(f"   - Эффективная ширина: ~{effective_width:.2f} с")