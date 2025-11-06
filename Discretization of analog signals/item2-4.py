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

# Задача 2.4. Сравнительный анализ импульсов

print("Задача 2.4. Сравнительный анализ импульсов")
print("=" * 60)

# Определение всех импульсов
def rectangular_pulse(t, tau):
    """Прямоугольный импульс"""
    return np.where((t >= -tau/2) & (t <= tau/2), 1, 0)

def triangular_pulse(t, tau):
    """Треугольный импульс"""
    return np.where(np.abs(t) <= tau/2, 1 - 2*np.abs(t)/tau, 0)

def gaussian_pulse(t, tau):
    """Гауссов импульс"""
    sigma_t = tau/4
    return np.exp(-(t**2) / (2 * sigma_t**2))

# Параметры для сравнения
tau = 1.0  # базовая длительность
t1, t2 = -2, 2  # временной интервал
t = np.linspace(t1, t2, 1000)

# Нормализация импульсов по энергии для честного сравнения
rect_norm = rectangular_pulse(t, tau)
triang_norm = triangular_pulse(t, tau) 
gauss_norm = gaussian_pulse(t, tau)

# Нормировка на одинаковую энергию
energy_rect = integrate.quad(lambda t: rectangular_pulse(t, tau)**2, -np.inf, np.inf)[0]
energy_triang = integrate.quad(lambda t: triangular_pulse(t, tau)**2, -np.inf, np.inf)[0]
energy_gauss = integrate.quad(lambda t: gaussian_pulse(t, tau)**2, -np.inf, np.inf)[0]

print(f"Энергии импульсов (до нормировки):")
print(f"Прямоугольный: {energy_rect:.4f}")
print(f"Треугольный: {energy_triang:.4f}")
print(f"Гауссов: {energy_gauss:.4f}")

# Нормированные импульсы с одинаковой энергией
rect_normalized = rect_norm / np.sqrt(energy_rect)
triang_normalized = triang_norm / np.sqrt(energy_triang)
gauss_normalized = gauss_norm / np.sqrt(energy_gauss)

# 1. Сравнение во временной области
plt.figure(figsize=[15, 12])

plt.subplot(3, 2, 1)
plt.plot(t, rect_normalized, 'r-', linewidth=2, label='Прямоугольный')
plt.plot(t, triang_normalized, 'g-', linewidth=2, label='Треугольный')
plt.plot(t, gauss_normalized, 'b-', linewidth=2, label='Гауссов')
plt.xlabel("Время, с")
plt.ylabel("Амплитуда (нормированная)")
plt.title("Сравнение импульсов во временной области\n(одинаковая энергия)")
plt.grid(True, alpha=0.3)
plt.legend()

# 2. Сравнение спектров
f = np.linspace(-15, 15, 1000)

# Аналитические спектры
X_rect = tau * np.sinc(f * tau) / np.sqrt(energy_rect)
X_triang = (tau/2) * (np.sinc(f * tau/2))**2 / np.sqrt(energy_triang)

sigma_t = tau/4
X_gauss = np.sqrt(2 * np.pi) * sigma_t * np.exp(-2 * (np.pi * f * sigma_t)**2) / np.sqrt(energy_gauss)

plt.subplot(3, 2, 2)
plt.plot(f, np.abs(X_rect), 'r-', linewidth=2, label='Прямоугольный')
plt.plot(f, np.abs(X_triang), 'g-', linewidth=2, label='Треугольный')
plt.plot(f, np.abs(X_gauss), 'b-', linewidth=2, label='Гауссов')
plt.xlabel("Частота, Гц")
plt.ylabel("|X(f)| (нормированный)")
plt.title("Сравнение спектров\n(одинаковая энергия)")
plt.grid(True, alpha=0.3)
plt.legend()

# 3. Спектры в логарифмическом масштабе
plt.subplot(3, 2, 3)
plt.semilogy(f, np.abs(X_rect), 'r-', linewidth=2, label='Прямоугольный')
plt.semilogy(f, np.abs(X_triang), 'g-', linewidth=2, label='Треугольный')
plt.semilogy(f, np.abs(X_gauss), 'b-', linewidth=2, label='Гауссов')
plt.xlabel("Частота, Гц")
plt.ylabel("|X(f)| (лог. шкала)")
plt.title("Спектры в логарифмическом масштабе")
plt.grid(True, alpha=0.3)
plt.legend()

# 4. Эффективная ширина спектра
print(f"\nАНАЛИЗ ЭФФЕКТИВНОЙ ШИРИНЫ СПЕКТРА:")
print("-" * 50)

# Ширина на уровне -3 дБ (0.707 от максимума)
level_3db = 0.707

def find_bandwidth(spectrum, freqs, level):
    """Находит ширину спектра на заданном уровне"""
    max_val = np.max(spectrum)
    target_val = level * max_val
    
    # Находим точки пересечения
    above_threshold = spectrum >= target_val
    if np.sum(above_threshold) == 0:
        return 0
    
    indices = np.where(above_threshold)[0]
    left_idx = indices[0]
    right_idx = indices[-1]
    
    return freqs[right_idx] - freqs[left_idx]

bw_rect = find_bandwidth(np.abs(X_rect), f, level_3db)
bw_triang = find_bandwidth(np.abs(X_triang), f, level_3db)
bw_gauss = find_bandwidth(np.abs(X_gauss), f, level_3db)

print(f"Ширина спектра на уровне -3 дБ:")
print(f"Прямоугольный: {bw_rect:.2f} Гц")
print(f"Треугольный: {bw_triang:.2f} Гц")
print(f"Гауссов: {bw_gauss:.2f} Гц")

# 5. Сравнение боковых лепестков
plt.subplot(3, 2, 4)
f_positive = f[f >= 0]
X_rect_positive = np.abs(X_rect)[f >= 0]
X_triang_positive = np.abs(X_triang)[f >= 0]
X_gauss_positive = np.abs(X_gauss)[f >= 0]

plt.semilogy(f_positive, X_rect_positive, 'r-', linewidth=2, label='Прямоугольный')
plt.semilogy(f_positive, X_triang_positive, 'g-', linewidth=2, label='Треугольный')
plt.semilogy(f_positive, X_gauss_positive, 'b-', linewidth=2, label='Гауссов')
plt.xlabel("Частота, Гц")
plt.ylabel("|X(f)| (лог. шкала)")
plt.title("Сравнение боковых лепестков")
plt.grid(True, alpha=0.3)
plt.legend()

# 6. Принцип неопределенности
print(f"\nПРИНЦИП НЕОПРЕДЕЛЕННОСТИ:")
print("-" * 50)

# Эффективная длительность во временной области
def find_effective_duration(signal, time, level):
    """Находит эффективную длительность на заданном уровне"""
    max_val = np.max(signal)
    target_val = level * max_val
    
    above_threshold = signal >= target_val
    if np.sum(above_threshold) == 0:
        return 0
    
    indices = np.where(above_threshold)[0]
    left_idx = indices[0]
    right_idx = indices[-1]
    
    return time[right_idx] - time[left_idx]

duration_rect = find_effective_duration(rect_normalized, t, level_3db)
duration_triang = find_effective_duration(triang_normalized, t, level_3db)
duration_gauss = find_effective_duration(gauss_normalized, t, level_3db)

print(f"Эффективная длительность на уровне -3 дБ:")
print(f"Прямоугольный: {duration_rect:.2f} с")
print(f"Треугольный: {duration_triang:.2f} с")
print(f"Гауссов: {duration_gauss:.2f} с")

# Произведение время-частота
uncertainty_rect = duration_rect * bw_rect
uncertainty_triang = duration_triang * bw_triang
uncertainty_gauss = duration_gauss * bw_gauss

print(f"\nПроизведение время-частота (Δt·Δf):")
print(f"Прямоугольный: {uncertainty_rect:.4f}")
print(f"Треугольный: {uncertainty_triang:.4f}")
print(f"Гауссов: {uncertainty_gauss:.4f}")
print(f"Теоретический минимум: {1/(4*np.pi):.4f}")

plt.subplot(3, 2, 5)
impulses = ['Прямоугольный', 'Треугольный', 'Гауссов']
uncertainties = [uncertainty_rect, uncertainty_triang, uncertainty_gauss]
colors = ['red', 'green', 'blue']

bars = plt.bar(impulses, uncertainties, color=colors, alpha=0.7)
plt.axhline(y=1/(4*np.pi), color='black', linestyle='--', label='Теоретический минимум')
plt.ylabel("Δt·Δf")
plt.title("Произведение время-частота")
plt.grid(True, alpha=0.3)
plt.legend()

# Добавляем значения на столбцы
for bar, value in zip(bars, uncertainties):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.3f}', ha='center', va='bottom')

# 7. Сравнение дискретизации
print(f"\nСРАВНЕНИЕ ДИСКРЕТИЗАЦИИ:")
print("-" * 50)

fs_values = [2/tau, 4/tau, 8/tau]
reconstruction_errors = {'rect': [], 'triang': [], 'gauss': []}

for fs in fs_values:
    T = 1/fs
    t_discrete = np.arange(t1, t2, T)
    
    # Дискретизация
    x_rect_disc = rectangular_pulse(t_discrete, tau)
    x_triang_disc = triangular_pulse(t_discrete, tau)
    x_gauss_disc = gaussian_pulse(t_discrete, tau)
    
    # Восстановление
    t_recon = np.linspace(t1, t2, 1000)
    x_rect_recon = np.zeros_like(t_recon)
    x_triang_recon = np.zeros_like(t_recon)
    x_gauss_recon = np.zeros_like(t_recon)
    
    for t_point, xr, xt, xg in zip(t_discrete, x_rect_disc, x_triang_disc, x_gauss_disc):
        sinc_base = np.sinc((t_recon - t_point) / T)
        x_rect_recon += xr * sinc_base
        x_triang_recon += xt * sinc_base
        x_gauss_recon += xg * sinc_base
    
    # Ошибка восстановления
    error_rect = np.sqrt(np.mean((rectangular_pulse(t_recon, tau) - x_rect_recon)**2))
    error_triang = np.sqrt(np.mean((triangular_pulse(t_recon, tau) - x_triang_recon)**2))
    error_gauss = np.sqrt(np.mean((gaussian_pulse(t_recon, tau) - x_gauss_recon)**2))
    
    reconstruction_errors['rect'].append(error_rect)
    reconstruction_errors['triang'].append(error_triang)
    reconstruction_errors['gauss'].append(error_gauss)
    
    print(f"fs = {fs:.1f} Гц:")
    print(f"  Прямоугольный: СКЗ ошибка = {error_rect:.6f}")
    print(f"  Треугольный: СКЗ ошибка = {error_triang:.6f}")
    print(f"  Гауссов: СКЗ ошибка = {error_gauss:.6f}")

plt.subplot(3, 2, 6)
plt.plot(fs_values, reconstruction_errors['rect'], 'ro-', label='Прямоугольный')
plt.plot(fs_values, reconstruction_errors['triang'], 'go-', label='Треугольный')
plt.plot(fs_values, reconstruction_errors['gauss'], 'bo-', label='Гауссов')
plt.xlabel("Частота дискретизации, Гц")
plt.ylabel("СКЗ ошибки восстановления")
plt.title("Ошибка восстановления при дискретизации")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# 8. Сравнение квантования
print(f"\nСРАВНЕНИЕ КВАНТОВАНИЯ:")
print("-" * 50)

# Дискретизированные сигналы с высокой частотой
fs_high = 20/tau
T_high = 1/fs_high
t_quant = np.linspace(-tau, tau, 200)

x_rect_quant = rectangular_pulse(t_quant, tau)
x_triang_quant = triangular_pulse(t_quant, tau)
x_gauss_quant = gaussian_pulse(t_quant, tau)

quant_levels = [4, 8, 16, 32]
quant_errors = {'rect': [], 'triang': [], 'gauss': []}

plt.figure(figsize=[15, 10])

for i, levels in enumerate(quant_levels):
    # Квантование
    x_rect_q = quantize_uniform(x_rect_quant, quant_min=0, quant_max=1, quant_level=levels)
    x_triang_q = quantize_uniform(x_triang_quant, quant_min=0, quant_max=1, quant_level=levels)
    x_gauss_q = quantize_uniform(x_gauss_quant, quant_min=0, quant_max=1, quant_level=levels)
    
    # Ошибки
    error_rect = np.sqrt(np.mean((x_rect_quant - x_rect_q)**2))
    error_triang = np.sqrt(np.mean((x_triang_quant - x_triang_q)**2))
    error_gauss = np.sqrt(np.mean((x_gauss_quant - x_gauss_q)**2))
    
    quant_errors['rect'].append(error_rect)
    quant_errors['triang'].append(error_triang)
    quant_errors['gauss'].append(error_gauss)
    
    print(f"L = {levels:2d}:")
    print(f"  Прямоугольный: СКЗ ошибка = {error_rect:.6f}")
    print(f"  Треугольный: СКЗ ошибка = {error_triang:.6f}")
    print(f"  Гауссов: СКЗ ошибка = {error_gauss:.6f}")

# График ошибок квантования
plt.subplot(2, 2, 1)
plt.plot(quant_levels, quant_errors['rect'], 'ro-', label='Прямоугольный')
plt.plot(quant_levels, quant_errors['triang'], 'go-', label='Треугольный')
plt.plot(quant_levels, quant_errors['gauss'], 'bo-', label='Гауссов')
plt.xscale('log')
plt.xlabel("Число уровней квантования")
plt.ylabel("СКЗ ошибки")
plt.title("Ошибка квантования")
plt.grid(True, alpha=0.3)
plt.legend()

# 9. Спектральные искажения от квантования
print(f"\nСПЕКТРАЛЬНЫЕ ИСКАЖЕНИЯ ОТ КВАНТОВАНИЯ:")
print("-" * 50)

# Квантованные сигналы для L=8
L_spectrum = 8
x_rect_q8 = quantize_uniform(x_rect_quant, quant_min=0, quant_max=1, quant_level=L_spectrum)
x_triang_q8 = quantize_uniform(x_triang_quant, quant_min=0, quant_max=1, quant_level=L_spectrum)
x_gauss_q8 = quantize_uniform(x_gauss_quant, quant_min=0, quant_max=1, quant_level=L_spectrum)

# Спектры ошибок
spectrum_rect_error = np.abs(np.fft.fftshift(np.fft.fft(x_rect_quant - x_rect_q8)))
spectrum_triang_error = np.abs(np.fft.fftshift(np.fft.fft(x_triang_quant - x_triang_q8)))
spectrum_gauss_error = np.abs(np.fft.fftshift(np.fft.fft(x_gauss_quant - x_gauss_q8)))
f_quant_spectrum = np.fft.fftshift(np.fft.fftfreq(len(x_rect_quant), T_high)) * fs_high

plt.subplot(2, 2, 2)
plt.semilogy(f_quant_spectrum, spectrum_rect_error, 'r-', label='Прямоугольный')
plt.semilogy(f_quant_spectrum, spectrum_triang_error, 'g-', label='Треугольный')
plt.semilogy(f_quant_spectrum, spectrum_gauss_error, 'b-', label='Гауссов')
plt.xlabel("Частота, Гц")
plt.ylabel("|E(f)|")
plt.title("Спектр ошибки квантования (L=8)")
plt.grid(True, alpha=0.3)
plt.legend()

# 10. Сводная таблица характеристик
print(f"\nСВОДНАЯ ТАБЛИЦА ХАРАКТЕРИСТИК:")
print("-" * 50)

characteristics = {
    'Параметр': ['Эффективная длительность', 'Ширина спектра', 'Δt·Δf', 'Боковые лепестки', 
                 'Скорость затухания', 'Ошибка дискретизации', 'Ошибка квантования'],
    'Прямоугольный': [f'{duration_rect:.3f} с', f'{bw_rect:.3f} Гц', f'{uncertainty_rect:.4f}',
                     'Большие', '~1/f', 'Высокая', 'Высокая'],
    'Треугольный': [f'{duration_triang:.3f} с', f'{bw_triang:.3f} Гц', f'{uncertainty_triang:.4f}',
                   'Средние', '~1/f²', 'Средняя', 'Средняя'],
    'Гауссов': [f'{duration_gauss:.3f} с', f'{bw_gauss:.3f} Гц', f'{uncertainty_gauss:.4f}',
               'Отсутствуют', '~exp(-f²)', 'Низкая', 'Низкая']
}

# Вывод таблицы
print(f"{'Параметр':<25} {'Прямоугольный':<15} {'Треугольный':<15} {'Гауссов':<15}")
print("-" * 70)
for i in range(len(characteristics['Параметр'])):
    param = characteristics['Параметр'][i]
    rect = characteristics['Прямоугольный'][i]
    triang = characteristics['Треугольный'][i]
    gauss = characteristics['Гауссов'][i]
    print(f"{param:<25} {rect:<15} {triang:<15} {gauss:<15}")

plt.subplot(2, 2, 3)
applications = {
    'Цифровая передача': [0.8, 0.6, 0.3],
    'Аудио обработка': [0.4, 0.7, 0.9],
    'Радиолокация': [0.3, 0.6, 0.9],
    'Медицинская визуализация': [0.2, 0.5, 0.95],
    'Телекоммуникации': [0.5, 0.8, 0.7]
}

app_names = list(applications.keys())
rect_scores = [app[0] for app in applications.values()]
triang_scores = [app[1] for app in applications.values()]
gauss_scores = [app[2] for app in applications.values()]

x_pos = np.arange(len(app_names))
width = 0.25

plt.bar(x_pos - width, rect_scores, width, label='Прямоугольный', alpha=0.7, color='red')
plt.bar(x_pos, triang_scores, width, label='Треугольный', alpha=0.7, color='green')
plt.bar(x_pos + width, gauss_scores, width, label='Гауссов', alpha=0.7, color='blue')

plt.xlabel("Область применения")
plt.ylabel("Оценка пригодности")
plt.title("Рекомендации по применению")
plt.xticks(x_pos, app_names, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# 12. Итоговый рейтинг
plt.subplot(2, 2, 4)
final_scores = {
    'Спектральная\nэффективность': [0.6, 0.8, 1.0],
    'Устойчивость к\nпомехам': [0.5, 0.7, 0.9],
    'Простота\nреализации': [1.0, 0.8, 0.6],
    'Качество\nвосстановления': [0.4, 0.7, 0.95]
}

score_names = list(final_scores.keys())
rect_final = [app[0] for app in final_scores.values()]
triang_final = [app[1] for app in final_scores.values()]
gauss_final = [app[2] for app in final_scores.values()]

x_pos_final = np.arange(len(score_names))

plt.plot(x_pos_final, rect_final, 'ro-', label='Прямоугольный', linewidth=2)
plt.plot(x_pos_final, triang_final, 'go-', label='Треугольный', linewidth=2)
plt.plot(x_pos_final, gauss_final, 'bo-', label='Гауссов', linewidth=2)

plt.xlabel("Критерий")
plt.ylabel("Оценка")
plt.title("Итоговый рейтинг по критериям")
plt.xticks(x_pos_final, score_names)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
