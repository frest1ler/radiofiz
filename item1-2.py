import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

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

# Задача 1.2. Дискретизация и квантование аудиосигнала

# Создадим тестовый аудиосигнал (синусоида 440 Гц - нота "Ля")
duration = 2.0  # секунды
f_audio = 440.0  # Гц (нота "Ля")
original_fs = 44100  # исходная частота дискретизации
t_audio = np.linspace(0, duration, int(original_fs * duration))
audio_signal = 0.5 * np.sin(2 * np.pi * f_audio * t_audio)

# Параметры дискретизации и квантования
downsample_factors = [2, 4, 8]  # коэффициенты понижения частоты дискретизации
quant_levels_list = [256, 64, 16, 4]  # уровни квантования

print("Задача 1.2: Дискретизация и квантование аудиосигнала")
print("=" * 60)

# Анализ исходного сигнала
print(f"Исходный сигнал:")
print(f"Длительность: {duration} с")
print(f"Частота дискретизации: {original_fs} Гц")
print(f"Частота сигнала: {f_audio} Гц")
print(f"Количество отсчетов: {len(audio_signal)}")
print(f"Динамический диапазон: [{np.min(audio_signal):.3f}, {np.max(audio_signal):.3f}]")

# 1. Исследование различных частот дискретизации
print(f"\n1) Исследование различных частот дискретизации:")

plt.figure(figsize=[15, 12])

# Исходный сигнал (высокая частота дискретизации)
plt.subplot(3, 2, 1)
segment_samples = 1000  # покажем только первые 1000 отсчетов для наглядности
t_segment = t_audio[:segment_samples] * 1000  # в миллисекундах
audio_segment = audio_signal[:segment_samples]
plt.plot(t_segment, audio_segment, 'b-', linewidth=1, label=f'fs={original_fs} Гц')
plt.xlabel("Время, мс")
plt.ylabel("Амплитуда")
plt.title("Исходный аудиосигнал (высокая частота дискретизации)")
plt.grid(True, alpha=0.3)
plt.legend()

# Понижение частоты дискретизации
for i, factor in enumerate(downsample_factors):
    fs_new = original_fs // factor
    # Децимация (прореживание)
    downsampled_signal = audio_signal[::factor]
    t_downsampled = t_audio[::factor]
    
    # Для сравнения берем тот же временной интервал
    compare_samples = min(segment_samples // factor, len(downsampled_signal))
    t_compare = t_downsampled[:compare_samples] * 1000
    signal_compare = downsampled_signal[:compare_samples]
    
    plt.subplot(3, 2, i+2)
    # Восстановленный сигнал (линейная интерполяция для наглядности)
    if len(t_compare) > 1:
        t_dense = np.linspace(t_compare[0], t_compare[-1], 1000)
        signal_dense = np.interp(t_dense, t_compare, signal_compare)
        plt.plot(t_dense, signal_dense, 'r-', alpha=0.7, linewidth=1, label='Восстановленный')
    
    plt.stem(t_compare, signal_compare, linefmt='g-', markerfmt='go', basefmt=' ', 
             label=f'Отсчеты (fs={fs_new} Гц)')
    plt.xlabel("Время, мс")
    plt.ylabel("Амплитуда")
    plt.title(f"Пониженная частота дискретизации (/{factor})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Анализ качества
    # Восстановление сигнала интерполяцией для сравнения
    t_original_dense = np.linspace(0, duration, len(audio_signal))
    reconstructed = np.interp(t_original_dense, t_downsampled, downsampled_signal)
    mse = np.mean((audio_signal - reconstructed) ** 2)
    
    print(f"fs = {fs_new:5d} Гц (/{factor}):")
    print(f"  Отсчетов: {len(downsampled_signal):6d}")
    print(f"  Теорема Котельникова: fs >= {2*f_audio} Гц -> {fs_new >= 2*f_audio}")
    print(f"  MSE ошибка: {mse:.6f}")

# 2. Исследование различного числа уровней квантования
print(f"\n2) Исследование различного числа уровней квантования:")

# Возьмем короткий сегмент для наглядности
segment_start = 1000
segment_end = 1200
audio_segment_for_quant = audio_signal[segment_start:segment_end]
t_segment_for_quant = t_audio[segment_start:segment_end] * 1000

plt.subplot(3, 2, 5)
for j, num_levels in enumerate(quant_levels_list):
    # Квантование сегмента
    quantized_segment = quantize_uniform(audio_segment_for_quant, 
                                       quant_min=-0.5, quant_max=0.5, 
                                       quant_level=num_levels)
    
    # Ошибка квантования
    quantization_error = np.abs(audio_segment_for_quant - quantized_segment)
    max_error = np.max(quantization_error)
    rms_error = np.sqrt(np.mean(quantization_error ** 2))
    
    plt.plot(t_segment_for_quant, quantized_segment, 
             label=f'L={num_levels}, err={max_error:.3f}')
    
    print(f"Уровней квантования: {num_levels:3d}")
    print(f"  Шаг квантования: {1.0/num_levels:.4f}")
    print(f"  Макс. ошибка: {max_error:.4f} (теор.: {0.5/num_levels:.4f})")
    print(f"  СКЗ ошибки: {rms_error:.4f}")

plt.plot(t_segment_for_quant, audio_segment_for_quant, 'k-', linewidth=2, 
         label='Исходный', alpha=0.7)
plt.xlabel("Время, мс")
plt.ylabel("Амплитуда")
plt.title("Квантование с различным числом уровней")
plt.grid(True, alpha=0.3)
plt.legend()

# 3. Совместное влияние дискретизации и квантования
print(f"\n3) Совместное влияние дискретизации и квантования:")

plt.subplot(3, 2, 6)
# Пример: умеренная дискретизация + различное квантование
fs_moderate = original_fs // 4
audio_moderate = audio_signal[::4]
t_moderate = t_audio[::4]

# Берем короткий сегмент
moderate_segment = audio_moderate[250:350]
t_moderate_segment = t_moderate[250:350] * 1000

quant_levels_demo = [256, 16]
colors = ['b', 'r']  # blue, red

for k, num_levels in enumerate(quant_levels_demo):
    quantized_moderate = quantize_uniform(moderate_segment, 
                                        quant_min=-0.5, quant_max=0.5,
                                        quant_level=num_levels)
    
    plt.stem(t_moderate_segment, quantized_moderate, 
             linefmt=colors[k], markerfmt=colors[k]+'o', basefmt=' ',
             label=f'L={num_levels}, fs={fs_moderate} Гц')

# Исходный сигнал для сравнения (интерполированный)
t_original_segment = t_audio[1000:1200] * 1000  # соответствующий временной интервал
audio_original_segment = audio_signal[1000:1200]
plt.plot(t_original_segment, audio_original_segment, 'g-', linewidth=2, 
         label='Исходный (fs=44100 Гц)', alpha=0.7)

plt.xlabel("Время, мс")
plt.ylabel("Амплитуда")
plt.title("Совместное влияние\nдискретизации и квантования")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# 4. Дополнительный анализ - спектральные характеристики
print(f"\n4) Спектральный анализ:")

plt.figure(figsize=[12, 8])

# Спектр исходного сигнала
plt.subplot(2, 2, 1)
frequencies = np.fft.fftfreq(len(audio_signal), 1/original_fs)
spectrum = np.abs(np.fft.fft(audio_signal))
positive_freq = frequencies[:len(frequencies)//2]
positive_spectrum = spectrum[:len(spectrum)//2]
plt.plot(positive_freq, positive_spectrum)
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда")
plt.title("Спектр исходного сигнала")
plt.grid(True, alpha=0.3)
plt.axvline(x=f_audio, color='r', linestyle='--', label=f'f={f_audio} Гц')
plt.legend()

# Спектр при пониженной частоте дискретизации
plt.subplot(2, 2, 2)
fs_low = original_fs // 8
audio_low = audio_signal[::8]
frequencies_low = np.fft.fftfreq(len(audio_low), 1/fs_low)
spectrum_low = np.abs(np.fft.fft(audio_low))
positive_freq_low = frequencies_low[:len(frequencies_low)//2]
positive_spectrum_low = spectrum_low[:len(spectrum_low)//2]
plt.plot(positive_freq_low, positive_spectrum_low)
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда")
plt.title(f"Спектр при fs={fs_low} Гц")
plt.grid(True, alpha=0.3)
plt.axvline(x=fs_low/2, color='r', linestyle='--', label=f'f_max={fs_low/2} Гц')
plt.legend()

# Спектр при сильном квантовании
plt.subplot(2, 2, 3)
audio_heavy_quant = quantize_uniform(audio_signal, quant_min=-0.5, quant_max=0.5, quant_level=4)
spectrum_quant = np.abs(np.fft.fft(audio_heavy_quant))
positive_spectrum_quant = spectrum_quant[:len(spectrum_quant)//2]
plt.plot(positive_freq, positive_spectrum_quant)
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда")
plt.title("Спектр при сильном квантовании (L=4)")
plt.grid(True, alpha=0.3)

# Сравнение SNR для разных уровней квантования
plt.subplot(2, 2, 4)
snr_values = []
for num_levels in quant_levels_list:
    quantized = quantize_uniform(audio_signal, quant_min=-0.5, quant_max=0.5, quant_level=num_levels)
    noise = audio_signal - quantized
    signal_power = np.mean(audio_signal**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    snr_values.append(snr)

plt.plot(quant_levels_list, snr_values, 'o-')
plt.xscale('log')
plt.xlabel("Число уровней квантования")
plt.ylabel("SNR, дБ")
plt.title("Зависимость SNR от числа уровней квантования")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Выводы и рекомендации
print(f"\nВЫВОДЫ:")
print("=" * 60)

print(f"\n1) ДИСКРЕТИЗАЦИЯ:")
print(f"   - Исходная fs={original_fs} Гц обеспечивает высокое качество")
print(f"   - При fs={original_fs//8} Гц появляются заметные искажения")
print(f"   - Теорема Котельникова выполняется для всех tested fs")

print(f"\n2) КВАНТОВАНИЕ:")
for num_levels in quant_levels_list:
    quantized = quantize_uniform(audio_signal, quant_min=-0.5, quant_max=0.5, quant_level=num_levels)
    noise = audio_signal - quantized
    signal_power = np.mean(audio_signal**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    print(f"   - L={num_levels:3d}: SNR={snr:6.1f} дБ, шаг={1.0/num_levels:.4f}")
