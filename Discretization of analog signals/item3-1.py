import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.io import wavfile

def quantize_uniform(x, quant_min=-1, quant_max=1, quant_level=5):
    """Uniform quantization approach"""
    x_normalize = (x-quant_min) * (quant_level-1) / (quant_max-quant_min)
    x_normalize[x_normalize > quant_level - 1] = quant_level - 1
    x_normalize[x_normalize < 0] = 0
    x_normalize_quant = np.around(x_normalize)
    x_quant = (x_normalize_quant) * (quant_max-quant_min) / (quant_level-1) + quant_min
    return x_quant

# Задача 3.1. Исследование речевого сигнала

print("Задача 3.1. Исследование речевого сигнала")
print("=" * 60)

# Создание тестового речеподобного сигнала
def create_speech_like_signal(duration=2.0, fs=44100):
    """
    Создание синтезированного речеподобного сигнала
    Имитирует основные характеристики речевого сигнала
    """
    t = np.linspace(0, duration, int(fs * duration))
    
    # Основные частоты речевого сигнала
    f_base = 120  # Основной тон (мужской голос)
    f_formant1 = 500   # Первая форманта
    f_formant2 = 1500  # Вторая форманта  
    f_formant3 = 2500  # Третья форманта
    
    # Создаем сложный сигнал с формантами
    signal = (0.5 * np.sin(2 * np.pi * f_base * t) +
              0.3 * np.sin(2 * np.pi * f_formant1 * t) +
              0.15 * np.sin(2 * np.pi * f_formant2 * t) +
              0.05 * np.sin(2 * np.pi * f_formant3 * t))
    
    # Добавляем шум для реалистичности
    noise = 0.02 * np.random.normal(0, 1, len(t))
    signal += noise
    
    # Амплитудная модуляция (имитация слогов)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)  # 4 Гц - частота слогов
    signal *= envelope
    
    # Нормализация
    signal = signal / np.max(np.abs(signal))
    
    return t, signal

# Параметры сигнала
duration = 2.0  # секунды
fs_original = 44100  # исходная частота дискретизации (качество CD)

# Создание речеподобного сигнала
t, speech_signal = create_speech_like_signal(duration, fs_original)

print(f"Параметры речевого сигнала:")
print(f"Длительность: {duration} с")
print(f"Частота дискретизации: {fs_original} Гц")
print(f"Количество отсчетов: {len(speech_signal)}")
print(f"Динамический диапазон: [{np.min(speech_signal):.3f}, {np.max(speech_signal):.3f}]")

# 1. Анализ исходного сигнала
plt.figure(figsize=[15, 10])

# Временная область
plt.subplot(3, 2, 1)
plt.plot(t, speech_signal, 'b-', linewidth=1, alpha=0.7)
plt.xlabel("Время, с")
plt.ylabel("Амплитуда")
plt.title("Речевой сигнал во временной области")
plt.grid(True, alpha=0.3)

# Спектр сигнала
plt.subplot(3, 2, 2)
frequencies = np.fft.fftfreq(len(speech_signal), 1/fs_original)
spectrum = np.abs(np.fft.fft(speech_signal))
positive_freq = frequencies[:len(frequencies)//2]
positive_spectrum = spectrum[:len(spectrum)//2]

plt.plot(positive_freq, positive_spectrum, 'r-', linewidth=1)
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда")
plt.title("Спектр речевого сигнала")
plt.grid(True, alpha=0.3)
plt.xlim(0, 5000)  # Ограничиваем до 5 кГц для наглядности

# Спектр в логарифмическом масштабе
plt.subplot(3, 2, 3)
plt.semilogy(positive_freq, positive_spectrum, 'g-', linewidth=1)
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда (лог. шкала)")
plt.title("Спектр речевого сигнала (лог. масштаб)")
plt.grid(True, alpha=0.3)
plt.xlim(0, 5000)

# Спектрограмма 
plt.subplot(3, 2, 4)
segment_length = 1024
overlap = 512
# Используем синтаксис для specgram
Pxx, freqs, bins, im = plt.specgram(speech_signal, NFFT=segment_length, 
                                    Fs=fs_original, noverlap=overlap, 
                                    cmap='viridis')
plt.xlabel("Время, с")
plt.ylabel("Частота, Гц")
plt.title("Спектрограмма речевого сигнала")
plt.colorbar(im, label='Интенсивность, дБ')
plt.ylim(0, 5000)

# Гистограмма амплитуд
plt.subplot(3, 2, 5)
plt.hist(speech_signal, bins=50, alpha=0.7, color='purple', edgecolor='black')
plt.xlabel("Амплитуда")
plt.ylabel("Частота")
plt.title("Распределение амплитуд речевого сигнала")
plt.grid(True, alpha=0.3)

# Автокорреляционная функция (для поиска основного тона)
plt.subplot(3, 2, 6)
autocorr = np.correlate(speech_signal, speech_signal, mode='full')
autocorr = autocorr[len(autocorr)//2:]
autocorr = autocorr / np.max(autocorr)  # Нормализация

plt.plot(t[:1000], autocorr[:1000], 'm-', linewidth=1)  # Первые 1000 отсчетов
plt.xlabel("Время, с")
plt.ylabel("Автокорреляция")
plt.title("Автокорреляционная функция")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Анализ характеристик речевого сигнала
print(f"\nАНАЛИЗ ХАРАКТЕРИСТИК РЕЧЕВОГО СИГНАЛА:")
print("-" * 50)

# Основные статистические характеристики
print(f"Статистические характеристики:")
print(f"Среднее значение: {np.mean(speech_signal):.6f}")
print(f"Стандартное отклонение: {np.std(speech_signal):.6f}")
print(f"Энергия сигнала: {np.sum(speech_signal**2):.6f}")

# Анализ спектральных характеристик
max_freq_idx = np.argmax(positive_spectrum[1:]) + 1  # Пропускаем DC компоненту
dominant_freq = positive_freq[max_freq_idx]
print(f"Доминирующая частота: {dominant_freq:.1f} Гц")

# Полоса частот, содержащая 95% энергии
cumulative_energy = np.cumsum(positive_spectrum**2)
total_energy = cumulative_energy[-1]
energy_95_idx = np.where(cumulative_energy >= 0.95 * total_energy)[0][0]
bandwidth_95 = positive_freq[energy_95_idx]
print(f"Полоса частот (95% энергии): {bandwidth_95:.1f} Гц")

# 2. Исследование дискретизации речевого сигнала
print(f"\nИССЛЕДОВАНИЕ ДИСКРЕТИЗАЦИИ РЕЧЕВОГО СИГНАЛА:")
print("-" * 50)

# Различные частоты дискретизации для тестирования
fs_test_values = [8000, 16000, 22050, 44100]  # Стандартные частоты дискретизации

plt.figure(figsize=[15, 12])

for i, fs_test in enumerate(fs_test_values):
    # Понижение частоты дискретизации
    downsample_factor = fs_original // fs_test
    speech_downsampled = speech_signal[::downsample_factor]
    t_downsampled = t[::downsample_factor]
    
    # Восстановление сигнала (для сравнения)
    t_recon = np.linspace(0, duration, len(speech_signal))
    speech_reconstructed = np.interp(t_recon, t_downsampled, speech_downsampled)
    
    # Ошибка восстановления
    reconstruction_error = np.abs(speech_signal - speech_reconstructed)
    
    plt.subplot(4, 3, i*3 + 1)
    plt.plot(t_recon, speech_signal, 'b-', alpha=0.7, linewidth=1, label='Исходный')
    plt.plot(t_downsampled, speech_downsampled, 'ro', markersize=2, label=f'fs={fs_test} Гц')
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда")
    plt.title(f"Дискретизация: fs={fs_test} Гц")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(4, 3, i*3 + 2)
    # Спектр пониженного сигнала
    spectrum_down = np.abs(np.fft.fft(speech_downsampled))
    freq_down = np.fft.fftfreq(len(speech_downsampled), 1/fs_test)
    pos_freq_down = freq_down[:len(freq_down)//2]
    pos_spectrum_down = spectrum_down[:len(spectrum_down)//2]
    
    plt.plot(pos_freq_down, pos_spectrum_down, 'r-', linewidth=1)
    plt.axvline(x=fs_test/2, color='gray', linestyle='--', alpha=0.7, label='f_Nyquist')
    plt.xlabel("Частота, Гц")
    plt.ylabel("Амплитуда")
    plt.title(f"Спектр (fs={fs_test} Гц)")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, fs_test)
    plt.legend()
    
    plt.subplot(4, 3, i*3 + 3)
    plt.plot(t_recon, reconstruction_error, 'g-', linewidth=1)
    plt.xlabel("Время, с")
    plt.ylabel("Ошибка")
    plt.title(f"Ошибка восстановления")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Анализ качества при различных fs
print(f"\nАНАЛИЗ КАЧЕСТВА ПРИ РАЗЛИЧНЫХ ЧАСТОТАХ ДИСКРЕТИЗАЦИИ:")
for fs_test in fs_test_values:
    downsample_factor = fs_original // fs_test
    speech_downsampled = speech_signal[::downsample_factor]
    t_downsampled = t[::downsample_factor]
    
    # Восстановление
    t_recon = np.linspace(0, duration, len(speech_signal))
    speech_reconstructed = np.interp(t_recon, t_downsampled, speech_downsampled)
    
    # Метрики качества
    mse = np.mean((speech_signal - speech_reconstructed)**2)
    snr = 10 * np.log10(np.mean(speech_signal**2) / mse) if mse > 0 else float('inf')
    
    print(f"fs = {fs_test:5d} Гц: MSE = {mse:.8f}, SNR = {snr:6.2f} дБ")

# 3. Исследование квантования речевого сигнала
print(f"\nИССЛЕДОВАНИЕ КВАНТОВАНИЯ РЕЧЕВОГО СИГНАЛА:")
print("-" * 50)

# Различные уровни квантования (битность)
bit_depths = [4, 8, 12, 16]  # бит на отсчет
quant_levels_list = [2**bits for bits in bit_depths]

plt.figure(figsize=[15, 10])

# Сегмент сигнала для детального анализа
segment_start = 10000
segment_end = 11000
speech_segment = speech_signal[segment_start:segment_end]
t_segment = t[segment_start:segment_end]

for i, (bits, levels) in enumerate(zip(bit_depths, quant_levels_list)):
    # Квантование всего сигнала
    speech_quantized = quantize_uniform(speech_signal, quant_min=-1, quant_max=1, 
                                      quant_level=levels)
    
    # Квантование сегмента для визуализации
    segment_quantized = quantize_uniform(speech_segment, quant_min=-1, quant_max=1, 
                                       quant_level=levels)
    
    # Ошибка квантования
    quantization_error = speech_signal - speech_quantized
    
    plt.subplot(4, 3, i*3 + 1)
    plt.plot(t_segment, speech_segment, 'b-', linewidth=2, label='Исходный', alpha=0.7)
    plt.step(t_segment, segment_quantized, 'r-', where='mid', label=f'{bits} бит', linewidth=1)
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда")
    plt.title(f"Квантование: {bits} бит ({levels} уровней)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(4, 3, i*3 + 2)
    # Спектр ошибки квантования
    spectrum_error = np.abs(np.fft.fft(quantization_error))
    pos_spectrum_error = spectrum_error[:len(spectrum_error)//2]
    
    plt.semilogy(positive_freq, pos_spectrum_error, 'g-', linewidth=1)
    plt.xlabel("Частота, Гц")
    plt.ylabel("|E(f)|")
    plt.title(f"Спектр ошибки квантования")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 5000)
    
    plt.subplot(4, 3, i*3 + 3)
    # Гистограмма ошибки
    plt.hist(quantization_error, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel("Ошибка квантования")
    plt.ylabel("Частота")
    plt.title(f"Распределение ошибки")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Анализ качества квантования
print(f"\nАНАЛИЗ КАЧЕСТВА КВАНТОВАНИЯ:")
quant_snr_values = []
quant_mse_values = []

for bits, levels in zip(bit_depths, quant_levels_list):
    speech_quantized = quantize_uniform(speech_signal, quant_min=-1, quant_max=1, 
                                      quant_level=levels)
    
    quantization_error = speech_signal - speech_quantized
    mse = np.mean(quantization_error**2)
    snr = 10 * np.log10(np.mean(speech_signal**2) / mse) if mse > 0 else float('inf')
    
    quant_snr_values.append(snr)
    quant_mse_values.append(mse)
    
    # Теоретический SNR для равномерного квантования: SNR ≈ 6.02·N + 1.76 дБ
    theoretical_snr = 6.02 * bits + 1.76
    
    print(f"{bits:2d} бит ({levels:5d} ур.): MSE = {mse:.8f}, SNR = {snr:6.2f} дБ, теор. = {theoretical_snr:5.2f} дБ")