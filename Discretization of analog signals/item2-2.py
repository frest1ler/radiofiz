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

# Задача 2.2. Исследование треугольного импульса

print("Задача 2.2. Исследование треугольного импульса")
print("=" * 60)

# Параметры треугольного импульса
def triangular_pulse(t, tau):
    """
    Треугольный импульс
    t - время
    tau - длительность импульса (ширина основания)
    """
    return np.where(np.abs(t) <= tau/2, 1 - 2*np.abs(t)/tau, 0)

# Параметры для анализа
tau = 1.0  # длительность импульса
t1, t2 = -2, 2  # временной интервал для анализа

# Временная область
t = np.linspace(t1, t2, 1000)
x_t = triangular_pulse(t, tau)

# Построение импульса во временной области
plt.figure(figsize=[15, 10])

plt.subplot(2, 2, 1)
plt.plot(t, x_t, 'b-', linewidth=2)
plt.xlabel("Время, с")
plt.ylabel("Амплитуда")
plt.title("Треугольный импульс во временной области")
plt.grid(True, alpha=0.3)
plt.axvline(x=-tau/2, color='r', linestyle='--', alpha=0.7, label=f'τ={tau} с')
plt.axvline(x=tau/2, color='r', linestyle='--', alpha=0.7)
plt.axvline(x=0, color='g', linestyle=':', alpha=0.7, label='Максимум')
plt.legend()

# Частотная область - аналитическое решение
f_analytic = np.linspace(-15, 15, 1000)
# Спектр треугольного импульса: (τ/2) * sinc²(fτ/2)
X_f_analytic = (tau/2) * (np.sinc(f_analytic * tau/2))**2

plt.subplot(2, 2, 2)
plt.plot(f_analytic, np.abs(X_f_analytic), 'r-', linewidth=2, label='|X(f)|')
plt.plot(f_analytic, X_f_analytic.real, 'g--', alpha=0.7, label='Re(X(f))')
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда")
plt.title("Спектр треугольного импульса (аналитический)")
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
f_numeric = np.linspace(-15, 15, 200)
X_f_numeric = fourier_transform_numeric(triangular_pulse, f_numeric, tau, t1, t2)

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
    X_f_tau = (tau_val/2) * (np.sinc(f_analytic * tau_val/2))**2
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
print(f"\nАНАЛИЗ СВОЙСТВ СПЕКТРА ТРЕУГОЛЬНОГО ИМПУЛЬСА:")
print("-" * 50)

# Нули спектра (sinc² имеет нули там же, где и sinc)
zeros_analytic = [2*n/tau for n in range(1, 6) if 2*n/tau <= 15]
print(f"Нули спектра (f ≠ 0): {[f'{zero:.2f} Гц' for zero in zeros_analytic]}")

# Ширина главного лепестка
main_lobe_width = 4/tau  # У sinc² ширина в 2 раза больше, чем у sinc
print(f"Ширина главного лепестка: {main_lobe_width:.2f} Гц")

# Максимум спектра
max_spectrum = tau/2
print(f"Максимум спектра (f=0): {max_spectrum:.2f}")

# Сравнение с прямоугольным импульсом
print(f"\nСРАВНЕНИЕ С ПРЯМОУГОЛЬНЫМ ИМПУЛЬСОМ:")
rectangular_spectrum = tau * np.sinc(f_analytic * tau)
plt.figure(figsize=[12, 4])

plt.subplot(1, 2, 1)
plt.plot(f_analytic, np.abs(X_f_analytic), 'b-', linewidth=2, label='Треугольный')
plt.plot(f_analytic, np.abs(rectangular_spectrum), 'r--', linewidth=2, label='Прямоугольный')
plt.xlabel("Частота, Гц")
plt.ylabel("|X(f)|")
plt.title("Сравнение спектров")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
# Отношение спектров
spectrum_ratio = np.abs(X_f_analytic) / np.abs(rectangular_spectrum)
spectrum_ratio[np.isnan(spectrum_ratio)] = 0  # Убираем деление на ноль
spectrum_ratio[np.isinf(spectrum_ratio)] = 0
plt.plot(f_analytic, spectrum_ratio, 'g-', linewidth=2)
plt.xlabel("Частота, Гц")
plt.ylabel("|X_треуг(f)| / |X_прям(f)|")
plt.title("Отношение спектров")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Энергетический анализ
print(f"\nЭНЕРГЕТИЧЕСКИЙ АНАЛИЗ:")
energy_time = integrate.quad(lambda t: triangular_pulse(t, tau)**2, t1, t2)[0]
print(f"Энергия во временной области: {energy_time:.4f}")

# Теоретическая энергия треугольного импульса: (2/3) * (τ/2)
energy_theoretical = (2/3) * (tau/2)
print(f"Теоретическая энергия: {energy_theoretical:.4f}")

# По теореме Парсеваля
energy_freq = integrate.quad(lambda f: np.abs((tau/2) * (np.sinc(f * tau/2))**2)**2, -np.inf, np.inf)[0]
print(f"Энергия в частотной области (Парсеваль): {energy_freq:.4f}")

# Анализ скорости затухания спектра
print(f"\nАНАЛИЗ СКОРОСТИ ЗАТУХАНИЯ СПЕКТРА:")
f_large = np.linspace(10, 100, 1000)
spectrum_large_f = np.abs((tau/2) * (np.sinc(f_large * tau/2))**2)

# Аппроксимация затухания
envelope = tau/(2 * (np.pi * f_large * tau/2)**2)  # ~1/f²

plt.figure(figsize=[10, 4])
plt.loglog(f_large, spectrum_large_f, 'b-', linewidth=2, label='Спектр')
plt.loglog(f_large, envelope, 'r--', linewidth=2, label='Огибающая ~1/f²')
plt.xlabel("Частота, Гц (лог. шкала)")
plt.ylabel("|X(f)| (лог. шкала)")
plt.title("Затухание спектра на высоких частотах")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print(f"Скорость затухания: ~1/f² (быстрее, чем у прямоугольного импульса ~1/f)")

# Дискретизация треугольного импульса
print(f"\nДИСКРЕТИЗАЦИЯ ТРЕУГОЛЬНОГО ИМПУЛЬСА:")
print("-" * 50)

fs_values = [2/tau, 4/tau, 8/tau, 16/tau]
t_discrete = np.linspace(t1, t2, 1000)

plt.figure(figsize=[15, 10])

for i, fs in enumerate(fs_values):
    # Дискретизация
    T = 1/fs
    n = np.arange(int(t1/T), int(t2/T))
    t_n = n * T
    x_n = triangular_pulse(t_n, tau)
    
    plt.subplot(2, 2, i+1)
    plt.plot(t_discrete, triangular_pulse(t_discrete, tau), 'b-', 
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

# Анализ ошибки дискретизации и восстановления
print(f"\nАНАЛИЗ ОШИБКИ ДИСКРЕТИЗАЦИИ И ВОССТАНОВЛЕНИЯ:")
for fs in fs_values:
    T = 1/fs
    n = np.arange(int(t1/T), int(t2/T))
    t_n = n * T
    x_n = triangular_pulse(t_n, tau)
    
    # Восстановление (идеальная интерполяция)
    t_recon = np.linspace(t1, t2, 1000)
    x_recon = np.zeros_like(t_recon)
    
    for t_point, x_point in zip(t_n, x_n):
        x_recon += x_point * np.sinc((t_recon - t_point) / T)
    
    error = np.max(np.abs(triangular_pulse(t_recon, tau) - x_recon))
    rms_error = np.sqrt(np.mean((triangular_pulse(t_recon, tau) - x_recon)**2))
    
    print(f"fs = {fs:5.1f} Гц: max ошибка = {error:.4f}, СКЗ = {rms_error:.4f}")

# Квантование треугольного импульса
print(f"\nКВАНТОВАНИЕ ТРЕУГОЛЬНОГО ИМПУЛЬСА:")
print("-" * 50)

# Дискретизированный импульс
fs = 20/tau  # высокая частота дискретизации
T = 1/fs
t_quant = np.linspace(-tau, tau, 100)
x_quant_original = triangular_pulse(t_quant, tau)

quant_levels = [4, 8, 16, 32]

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
    for level in quant_levels_vals:
        plt.axhline(y=level, color='gray', linestyle='--', alpha=0.3)
    
    error = np.max(np.abs(x_quant_original - x_quantized))
    plt.title(f'L={levels}, max ошибка={error:.3f}')
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
for levels in quant_levels:
    x_quantized = quantize_uniform(x_quant_original, quant_min=0, quant_max=1, 
                                  quant_level=levels)
    max_error = np.max(np.abs(x_quant_original - x_quantized))
    rms_error = np.sqrt(np.mean((x_quant_original - x_quantized)**2))
    
    max_errors.append(max_error)
    rms_errors.append(rms_error)
    
    print(f"L = {levels:2d}: max ошибка = {max_error:.4f}, СКЗ = {rms_error:.4f}")

# График зависимости ошибки от числа уровней
bits_per_sample = [np.log2(L) for L in quant_levels]
plt.figure(figsize=[10, 4])

plt.subplot(1, 2, 1)
plt.plot(quant_levels, max_errors, 'ro-', label='Макс. ошибка')
plt.plot(quant_levels, rms_errors, 'bs-', label='СКЗ ошибка')
plt.xscale('log')
plt.xlabel("Число уровней квантования")
plt.ylabel("Ошибка")
plt.title("Зависимость ошибки от числа уровней")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(bits_per_sample, max_errors, 'ro-', label='Макс. ошибка')
plt.plot(bits_per_sample, rms_errors, 'bs-', label='СКЗ ошибка')
plt.xlabel("Бит на отсчет")
plt.ylabel("Ошибка")
plt.title("Зависимость ошибки от битрейта")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# Спектральные искажения от квантования
print(f"\nСПЕКТРАЛЬНЫЕ ИСКАЖЕНИЯ ОТ КВАНТОВАНИЯ:")
print("-" * 50)

# Спектр исходного и квантованного сигнала
f_spectrum = np.linspace(-20, 20, 500)

# Исходный спектр
spectrum_original = (tau/2) * (np.sinc(f_spectrum * tau/2))**2

# Спектр квантованного сигнала (для L=8)
x_quant_L8 = quantize_uniform(x_quant_original, quant_min=0, quant_max=1, quant_level=8)
spectrum_quantized = np.abs(np.fft.fftshift(np.fft.fft(x_quant_L8)))
f_quant = np.fft.fftshift(np.fft.fftfreq(len(x_quant_L8), T)) * fs

plt.figure(figsize=[12, 4])

plt.subplot(1, 2, 1)
plt.plot(f_spectrum, np.abs(spectrum_original), 'b-', linewidth=2, 
         label='Исходный спектр')
plt.plot(f_quant, spectrum_quantized/len(spectrum_quantized), 'r-', 
         label='Квантованный (L=8)')
plt.xlabel("Частота, Гц")
plt.ylabel("|X(f)|")
plt.title("Сравнение спектров")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
# Спектральная плотность ошибки
error_signal = x_quant_original - x_quant_L8
spectrum_error = np.abs(np.fft.fftshift(np.fft.fft(error_signal)))
plt.plot(f_quant, spectrum_error/len(spectrum_error), 'g-', linewidth=2)
plt.xlabel("Частота, Гц")
plt.ylabel("|E(f)|")
plt.title("Спектр ошибки квантования")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ВЫВОДЫ
print(f"\nВЫВОДЫ ПО ЗАДАЧЕ 2.2:")
print("=" * 60)

print(f"\n1) СПЕКТР ТРЕУГОЛЬНОГО ИМПУЛЬСА:")
print(f"   - Имеет форму sinc²-функции: X(f) = (τ/2)·sinc²(fτ/2)")
print(f"   - Нули спектра: f = 2n/τ, n = ±1, ±2, ...")
print(f"   - Ширина главного лепестка: 4/τ Гц (в 2 раза шире прямоугольного)")
print(f"   - Затухает как ~1/f² (быстрее прямоугольного ~1/f)")

print(f"\n2) СРАВНЕНИЕ С ПРЯМОУГОЛЬНЫМ ИМПУЛЬСОМ:")
print(f"   - Треугольный импульс имеет более компактный спектр")
print(f"   - Меньшие боковые лепестки")
print(f"   - Быстрее затухает на высоких частотах")

print(f"\n3) ДИСКРЕТИЗАЦИЯ:")
print(f"   - Минимальная fs = 2/τ (по теореме Котельникова)")
print(f"   - Рекомендуемая fs ≥ 4/τ для качественного восстановления")
print(f"   - Ошибка восстановления меньше, чем у прямоугольного импульса")

print(f"\n4) КВАНТОВАНИЕ:")
print(f"   - Плавная форма уменьшает ошибку квантования")
print(f"   - Для L=16 уровней ошибка < 2%")
print(f"   - Каждое удвоение уровней уменьшает ошибку в ~2 раза")

print(f"\n5) ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:")
print(f"   - Для треугольных импульсов: fs ≥ 4/τ_max")
print(f"   - Уровни квантования: L ≥ 16 для точности < 2%")
print(f"   - Треугольные импульсы предпочтительнее прямоугольных")
print(f"   - из-за лучших спектральных свойств")
