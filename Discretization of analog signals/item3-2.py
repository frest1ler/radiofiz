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

# Задача 3.2. Исследование музыкального сигнала

print("Задача 3.2. Исследование музыкального сигнала")
print("=" * 60)

# Создание тестового музыкального сигнала
def create_music_like_signal(duration=3.0, fs=44100):
    """
    Создание синтезированного музыкального сигнала
    Имитирует основные характеристики музыкального сигнала
    """
    t = np.linspace(0, duration, int(fs * duration))
    
    # Аккорд (до-мажор: C4, E4, G4)
    f_C4 = 261.63  # До 4-й октавы
    f_E4 = 329.63  # Ми 4-й октавы
    f_G4 = 392.00  # Соль 4-й октавы
    
    # Основной музыкальный сигнал (аккорд)
    chord = (0.4 * np.sin(2 * np.pi * f_C4 * t) +
             0.3 * np.sin(2 * np.pi * f_E4 * t) + 
             0.2 * np.sin(2 * np.pi * f_G4 * t))
    
    # Добавляем гармоники для богатого тембра
    harmonics = (0.1 * np.sin(2 * np.pi * 2 * f_C4 * t) +
                 0.05 * np.sin(2 * np.pi * 3 * f_C4 * t) +
                 0.05 * np.sin(2 * np.pi * 2 * f_E4 * t) +
                 0.05 * np.sin(2 * np.pi * 2 * f_G4 * t))
    
    # Басовая линия (С3)
    f_C3 = 130.81  # До 3-й октавы
    bass = 0.2 * np.sin(2 * np.pi * f_C3 * t)
    
    # Ударные (имитация барабана)
    drum_envelope = np.exp(-5 * t) * (t % 0.5 < 0.05)
    drum = 0.3 * drum_envelope * np.random.normal(0, 1, len(t))
    
    # Объединяем все компоненты
    music_signal = chord + harmonics + bass + drum
    
    # Амплитудная огибающая (имитация музыкальной фразы)
    envelope = 0.5 + 0.3 * np.sin(2 * np.pi * 0.5 * t) + 0.2 * np.sin(2 * np.pi * 2 * t)
    music_signal *= envelope
    
    # Нормализация
    music_signal = music_signal / np.max(np.abs(music_signal))
    
    return t, music_signal

# Параметры сигнала
duration = 3.0  # секунды
fs_original = 44100  # исходная частота дискретизации (качество CD)

# Создание музыкального сигнала
t, music_signal = create_music_like_signal(duration, fs_original)

print(f"Параметры музыкального сигнала:")
print(f"Длительность: {duration} с")
print(f"Частота дискретизации: {fs_original} Гц")
print(f"Количество отсчетов: {len(music_signal)}")
print(f"Динамический диапазон: [{np.min(music_signal):.3f}, {np.max(music_signal):.3f}]")

# 1. Анализ исходного сигнала
plt.figure(figsize=[15, 12])

# Временная область
plt.subplot(3, 2, 1)
plt.plot(t, music_signal, 'b-', linewidth=1, alpha=0.7)
plt.xlabel("Время, с")
plt.ylabel("Амплитуда")
plt.title("Музыкальный сигнал во временной области")
plt.grid(True, alpha=0.3)

# Спектр сигнала
plt.subplot(3, 2, 2)
frequencies = np.fft.fftfreq(len(music_signal), 1/fs_original)
spectrum = np.abs(np.fft.fft(music_signal))
positive_freq = frequencies[:len(frequencies)//2]
positive_spectrum = spectrum[:len(spectrum)//2]

plt.plot(positive_freq, positive_spectrum, 'r-', linewidth=1)
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда")
plt.title("Спектр музыкального сигнала")
plt.grid(True, alpha=0.3)
plt.xlim(0, 10000)  # Ограничиваем до 10 кГц для наглядности

# Отметим основные частоты аккорда
frequencies_notes = [130.81, 261.63, 329.63, 392.00]  # C3, C4, E4, G4
note_names = ['C3', 'C4', 'E4', 'G4']
colors = ['red', 'blue', 'green', 'purple']

for freq, name, color in zip(frequencies_notes, note_names, colors):
    plt.axvline(x=freq, color=color, linestyle='--', alpha=0.7, label=name)

plt.legend()

# Спектр в логарифмическом масштабе
plt.subplot(3, 2, 3)
plt.semilogy(positive_freq, positive_spectrum, 'g-', linewidth=1)
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда (лог. шкала)")
plt.title("Спектр музыкального сигнала (лог. масштаб)")
plt.grid(True, alpha=0.3)
plt.xlim(0, 10000)

for freq, name, color in zip(frequencies_notes, note_names, colors):
    plt.axvline(x=freq, color=color, linestyle='--', alpha=0.7)

# Спектрограмма
plt.subplot(3, 2, 4)
segment_length = 2048  # Больше для лучшего частотного разрешения
overlap = 1024
Pxx, freqs, bins, im = plt.specgram(music_signal, NFFT=segment_length, 
                                    Fs=fs_original, noverlap=overlap, 
                                    cmap='hot')
plt.xlabel("Время, с")
plt.ylabel("Частота, Гц")
plt.title("Спектрограмма музыкального сигнала")
plt.colorbar(im, label='Интенсивность, дБ')
plt.ylim(0, 5000)

# Гистограмма амплитуд
plt.subplot(3, 2, 5)
plt.hist(music_signal, bins=50, alpha=0.7, color='purple', edgecolor='black')
plt.xlabel("Амплитуда")
plt.ylabel("Частота")
plt.title("Распределение амплитуд музыкального сигнала")
plt.grid(True, alpha=0.3)

# Крест-корреляция с эталонными частотами
plt.subplot(3, 2, 6)
# Создаем эталонные синусоиды для нот
correlation_values = []
for freq in frequencies_notes:
    reference = np.sin(2 * np.pi * freq * t[:1000])  # Первые 1000 отсчетов
    correlation = np.correlate(music_signal[:1000], reference, mode='valid')
    correlation_values.append(np.max(np.abs(correlation)))

plt.bar(note_names, correlation_values, color=colors, alpha=0.7)
plt.xlabel("Нота")
plt.ylabel("Корреляция")
plt.title("Корреляция с эталонными нотами")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Анализ характеристик музыкального сигнала
print(f"\nАНАЛИЗ ХАРАКТЕРИСТИК МУЗЫКАЛЬНОГО СИГНАЛА:")
print("-" * 50)

# Основные статистические характеристики
print(f"Статистические характеристики:")
print(f"Среднее значение: {np.mean(music_signal):.6f}")
print(f"Стандартное отклонение: {np.std(music_signal):.6f}")
print(f"Энергия сигнала: {np.sum(music_signal**2):.6f}")
print(f"Пик-фактор: {np.max(np.abs(music_signal)) / np.std(music_signal):.2f}")

# Анализ спектральных характеристик
print(f"\nСпектральные характеристики:")
# Полоса частот, содержащая 95% энергии
cumulative_energy = np.cumsum(positive_spectrum**2)
total_energy = cumulative_energy[-1]
energy_95_idx = np.where(cumulative_energy >= 0.95 * total_energy)[0][0]
bandwidth_95 = positive_freq[energy_95_idx]
print(f"Полоса частот (95% энергии): {bandwidth_95:.1f} Гц")

# Полоса частот, содержащая 99% энергии
energy_99_idx = np.where(cumulative_energy >= 0.99 * total_energy)[0][0]
bandwidth_99 = positive_freq[energy_99_idx]
print(f"Полоса частот (99% энергии): {bandwidth_99:.1f} Гц")

# Центральная частота спектра
spectral_centroid = np.sum(positive_freq * positive_spectrum) / np.sum(positive_spectrum)
print(f"Спектральный центроид: {spectral_centroid:.1f} Гц")

# 2. Исследование дискретизации музыкального сигнала
print(f"\nИССЛЕДОВАНИЕ ДИСКРЕТИЗАЦИИ МУЗЫКАЛЬНОГО СИГНАЛА:")
print("-" * 50)

# Различные частоты дискретизации для тестирования
fs_test_values = [8000, 16000, 22050, 44100, 48000, 96000]

plt.figure(figsize=[15, 15])

for i, fs_test in enumerate(fs_test_values):
    if fs_test <= fs_original:  # Только понижение частоты
        # Понижение частоты дискретизации
        downsample_factor = fs_original // fs_test
        music_downsampled = music_signal[::downsample_factor]
        t_downsampled = t[::downsample_factor]
        
        # Восстановление сигнала (для сравнения)
        t_recon = np.linspace(0, duration, len(music_signal))
        music_reconstructed = np.interp(t_recon, t_downsampled, music_downsampled)
        
        # Ошибка восстановления
        reconstruction_error = np.abs(music_signal - music_reconstructed)
        
        plt.subplot(6, 3, i*3 + 1)
        # Показываем короткий сегмент для наглядности
        seg_start = 50000
        seg_end = 51000
        plt.plot(t_recon[seg_start:seg_end], music_signal[seg_start:seg_end], 
                'b-', alpha=0.7, linewidth=1, label='Исходный')
        plt.plot(t_downsampled, music_downsampled, 'ro', markersize=2, 
                label=f'fs={fs_test} Гц')
        plt.xlabel("Время, с")
        plt.ylabel("Амплитуда")
        plt.title(f"Дискретизация: fs={fs_test} Гц")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(6, 3, i*3 + 2)
        # Спектр пониженного сигнала
        spectrum_down = np.abs(np.fft.fft(music_downsampled))
        freq_down = np.fft.fftfreq(len(music_downsampled), 1/fs_test)
        pos_freq_down = freq_down[:len(freq_down)//2]
        pos_spectrum_down = spectrum_down[:len(spectrum_down)//2]
        
        plt.plot(pos_freq_down, pos_spectrum_down, 'r-', linewidth=1)
        plt.axvline(x=fs_test/2, color='gray', linestyle='--', alpha=0.7, label='f_Nyquist')
        plt.xlabel("Частота, Гц")
        plt.ylabel("Амплитуда")
        plt.title(f"Спектр (fs={fs_test} Гц)")
        plt.grid(True, alpha=0.3)
        plt.xlim(0, min(fs_test, 20000))
        plt.legend()
        
        plt.subplot(6, 3, i*3 + 3)
        plt.plot(t_recon, reconstruction_error, 'g-', linewidth=1)
        plt.xlabel("Время, с")
        plt.ylabel("Ошибка")
        plt.title(f"Ошибка восстановления")
        plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Анализ качества при различных fs
print(f"\nАНАЛИЗ КАЧЕСТВА ПРИ РАЗЛИЧНЫХ ЧАСТОТАХ ДИСКРЕТИЗАЦИИ:")
quality_metrics = []

for fs_test in fs_test_values:
    if fs_test <= fs_original:
        downsample_factor = fs_original // fs_test
        music_downsampled = music_signal[::downsample_factor]
        t_downsampled = t[::downsample_factor]
        
        # Восстановление
        t_recon = np.linspace(0, duration, len(music_signal))
        music_reconstructed = np.interp(t_recon, t_downsampled, music_downsampled)
        
        # Метрики качества
        mse = np.mean((music_signal - music_reconstructed)**2)
        snr = 10 * np.log10(np.mean(music_signal**2) / mse) if mse > 0 else float('inf')
        
        # Субъективная оценка (на основе полосы пропускания)
        if fs_test >= 44100:
            subjective_quality = "Отличное"
        elif fs_test >= 22050:
            subjective_quality = "Хорошее" 
        elif fs_test >= 16000:
            subjective_quality = "Удовлетворительное"
        else:
            subjective_quality = "Плохое"
            
        quality_metrics.append((fs_test, mse, snr, subjective_quality))
        print(f"fs = {fs_test:5d} Гц: MSE = {mse:.8f}, SNR = {snr:6.2f} дБ, Качество: {subjective_quality}")

# 3. Исследование квантования музыкального сигнала
print(f"\nИССЛЕДОВАНИЕ КВАНТОВАНИЯ МУЗЫКАЛЬНОГО СИГНАЛА:")
print("-" * 50)

# Различные уровни квантования (битность)
bit_depths = [8, 12, 16, 20, 24]  # бит на отсчет
quant_levels_list = [2**bits for bits in bit_depths]

plt.figure(figsize=[15, 12])

# Сегмент сигнала для детального анализа
segment_start = 50000
segment_end = 51000
music_segment = music_signal[segment_start:segment_end]
t_segment = t[segment_start:segment_end]

quant_analysis_results = []

for i, (bits, levels) in enumerate(zip(bit_depths, quant_levels_list)):
    # Квантование всего сигнала
    music_quantized = quantize_uniform(music_signal, quant_min=-1, quant_max=1, 
                                     quant_level=levels)
    
    # Квантование сегмента для визуализации
    segment_quantized = quantize_uniform(music_segment, quant_min=-1, quant_max=1, 
                                      quant_level=levels)
    
    # Ошибка квантования
    quantization_error = music_signal - music_quantized
    
    # Метрики качества
    mse = np.mean(quantization_error**2)
    snr = 10 * np.log10(np.mean(music_signal**2) / mse)
    theoretical_snr = 6.02 * bits + 1.76
    
    quant_analysis_results.append((bits, levels, mse, snr, theoretical_snr))
    
    plt.subplot(5, 3, i*3 + 1)
    plt.plot(t_segment, music_segment, 'b-', linewidth=2, label='Исходный', alpha=0.7)
    plt.step(t_segment, segment_quantized, 'r-', where='mid', label=f'{bits} бит', linewidth=1)
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда")
    plt.title(f"Квантование: {bits} бит ({levels} ур.)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(5, 3, i*3 + 2)
    # Спектр ошибки квантования
    spectrum_error = np.abs(np.fft.fft(quantization_error))
    pos_spectrum_error = spectrum_error[:len(spectrum_error)//2]
    
    plt.semilogy(positive_freq, pos_spectrum_error, 'g-', linewidth=1)
    plt.xlabel("Частота, Гц")
    plt.ylabel("|E(f)|")
    plt.title(f"Спектр ошибки квантования")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 20000)
    
    plt.subplot(5, 3, i*3 + 3)
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
print(f"{'Битность':<8} {'Уровни':<10} {'MSE':<12} {'SNR (дБ)':<10} {'Теор. SNR':<10} {'Разница':<8}")
print("-" * 65)
for bits, levels, mse, snr, theoretical_snr in quant_analysis_results:
    difference = snr - theoretical_snr
    print(f"{bits:<8} {levels:<10} {mse:<12.8f} {snr:<10.2f} {theoretical_snr:<10.2f} {difference:<8.2f}")

# 4. Совместное влияние дискретизации и квантования
print(f"\nСОВМЕСТНОЕ ВЛИЯНИЕ ДИСКРЕТИЗАЦИИ И КВАНТОВАНИЯ:")
print("-" * 50)

# Комбинации параметров для различных стандартов качества
quality_standards = [
    ("Телефон", 8000, 8),
    ("Радио", 22050, 12), 
    ("CD", 44100, 16),
    ("Студия", 96000, 24)
]

plt.figure(figsize=[12, 8])

for i, (standard_name, fs_std, bits_std) in enumerate(quality_standards):
    if fs_std <= fs_original:
        # Дискретизация
        downsample_factor = fs_original // fs_std
        music_disc = music_signal[::downsample_factor]
        t_disc = t[::downsample_factor]
        
        # Квантование
        levels_std = 2**bits_std
        music_disc_quant = quantize_uniform(music_disc, quant_min=-1, quant_max=1, 
                                          quant_level=levels_std)
        
        # Восстановление для сравнения
        t_recon = np.linspace(0, duration, len(music_signal))
        music_recon = np.interp(t_recon, t_disc, music_disc_quant)
        
        # Общая ошибка
        total_error = music_signal - music_recon
        
        plt.subplot(2, 2, i+1)
        # Сегмент для визуализации
        seg_start_vis = 50000
        seg_end_vis = 50500
        plt.plot(t_recon[seg_start_vis:seg_end_vis], music_signal[seg_start_vis:seg_end_vis], 
                'b-', linewidth=2, label='Исходный', alpha=0.7)
        plt.plot(t_recon[seg_start_vis:seg_end_vis], music_recon[seg_start_vis:seg_end_vis], 
                'r-', linewidth=1, label=f'{standard_name}\nfs={fs_std} Гц, {bits_std} бит')
        plt.xlabel("Время, с")
        plt.ylabel("Амплитуда")
        plt.title(f"Стандарт: {standard_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()

plt.tight_layout()
plt.show()

# Анализ комбинированного влияния
print(f"\nАНАЛИЗ КОМБИНИРОВАННОГО ВЛИЯНИЯ:")
print(f"{'Стандарт':<10} {'fs (Гц)':<8} {'Битность':<10} {'MSE':<12} {'SNR (дБ)':<10}")
print("-" * 55)
for standard_name, fs_std, bits_std in quality_standards:
    if fs_std <= fs_original:
        downsample_factor = fs_original // fs_std
        music_disc = music_signal[::downsample_factor]
        t_disc = t[::downsample_factor]
        
        levels_std = 2**bits_std
        music_disc_quant = quantize_uniform(music_disc, quant_min=-1, quant_max=1, 
                                          quant_level=levels_std)
        
        t_recon = np.linspace(0, duration, len(music_signal))
        music_recon = np.interp(t_recon, t_disc, music_disc_quant)
        
        total_error = music_signal - music_recon
        mse_total = np.mean(total_error**2)
        snr_total = 10 * np.log10(np.mean(music_signal**2) / mse_total)
        
        print(f"{standard_name:<10} {fs_std:<8} {bits_std:<10} {mse_total:<12.8f} {snr_total:<10.2f}")

# 5. Сравнение с речевым сигналом (из задачи 3.1)
print(f"\nСРАВНЕНИЕ МУЗЫКАЛЬНОГО И РЕЧЕВОГО СИГНАЛОВ:")
print("-" * 50)

# Создаем речевой сигнал для сравнения
def create_speech_for_comparison(duration=3.0, fs=44100):
    """Создание речевого сигнала для сравнения"""
    t_comp = np.linspace(0, duration, int(fs * duration))
    
    # Простой речевой сигнал с формантами
    f_base = 120
    f_formants = [500, 1500, 2500]
    
    speech = 0.6 * np.sin(2 * np.pi * f_base * t_comp)
    for i, f_formant in enumerate(f_formants):
        speech += (0.3/(i+1)) * np.sin(2 * np.pi * f_formant * t_comp)
    
    return speech / np.max(np.abs(speech))

speech_comp = create_speech_for_comparison(duration, fs_original)

# Сравнительные характеристики
comparison_data = {
    'Характеристика': ['Полоса 95% энергии', 'Полоса 99% энергии', 'Спектр. центроид', 
                      'Станд. отклонение', 'Пик-фактор', 'Динамический диапазон'],
    'Музыка': [f"{bandwidth_95:.1f} Гц", f"{bandwidth_99:.1f} Гц", 
               f"{spectral_centroid:.1f} Гц", f"{np.std(music_signal):.4f}",
               f"{np.max(np.abs(music_signal))/np.std(music_signal):.2f}",
               f"{20*np.log10(np.max(np.abs(music_signal))/np.std(music_signal)):.1f} дБ"],
    'Речь': ['-', '-', '-', f"{np.std(speech_comp):.4f}",
            f"{np.max(np.abs(speech_comp))/np.std(speech_comp):.2f}",
            f"{20*np.log10(np.max(np.abs(speech_comp))/np.std(speech_comp)):.1f} дБ"]
}

print(f"{'Характеристика':<25} {'Музыка':<15} {'Речь':<15}")
print("-" * 55)
for i in range(len(comparison_data['Характеристика'])):
    char = comparison_data['Характеристика'][i]
    music_val = comparison_data['Музыка'][i]
    speech_val = comparison_data['Речь'][i]
    print(f"{char:<25} {music_val:<15} {speech_val:<15}")

# ВЫВОДЫ
print(f"\nХАРАКТЕРИСТИКИ СИГНАЛА:")
print(f"   - Широкая полоса частот (до {bandwidth_99:.0f} Гц для 99% энергии)")
print(f"   - Наличие гармонических структур (аккорды, обертоны)")