import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# Декодирование WAV файла clarinet.wav из папки audio_files
audio_file = 'audio_files/clarinet.wav'

# Проверяем существование файла
if not os.path.exists(audio_file):
    print(f"Ошибка: Файл {audio_file} не найден!")
    print("Убедитесь, что файл находится в папке audio_files")
    exit()

# Чтение аудиофайла
fs, x = wavfile.read(audio_file)

print("=" * 60)
print("ЗАДАЧА 1.2: ДЕКОДИРОВАНИЕ WAV ФАЙЛА (ВАРИАНТ 1)")
print("=" * 60)

# 1) Определение параметров файла
print("\n1) Параметры WAV файла clarinet.wav:")
print(f"   Частота дискретизации: {fs} Гц")
print(f"   Тип данных (dtype): {x.dtype}")
print(f"   Размер массива: {x.size} отсчетов")

# Определение битной глубины
if x.dtype == np.int16:
    bit_depth = 16
    quant_levels = 2**16
    print(f"   Битная глубина: {bit_depth} бит")
    print(f"   Количество уровней квантования: {quant_levels}")
elif x.dtype == np.int32:
    bit_depth = 32
    quant_levels = 2**32
    print(f"   Битная глубина: {bit_depth} бит")
    print(f"   Количество уровней квантования: {quant_levels}")
else:
    bit_depth = 16
    quant_levels = 2**16
    print(f"   Битная глубина: {bit_depth} бит (предположительно)")
    print(f"   Количество уровней квантования: {quant_levels}")

# Если аудио стерео, берем только один канал
if len(x.shape) > 1:
    x_mono = x[:, 0]
    print(f"   Файл стерео, используется левый канал")
else:
    x_mono = x
    print(f"   Файл моно")

# 2) Анализ расстояния между отсчетами
delta_t = 1 / fs
print(f"\n2) Анализ расстояния между отсчетами:")
print(f"   Расстояние между отсчетами: Δt = 1/fs = {delta_t:.6f} с")
print(f"   На графике видно равномерное распределение отсчетов с интервалом {delta_t*1000:.3f} мс")

# 3) Построение графиков с правильной осью времени
print(f"\n3) График с осью времени от начала файла:")

# Выбор наблюдаемого диапазона (как в методичке)
x1 = x_mono[8000:10000]  # выбор наблюдаемого диапазона

# Создаем ось времени в секундах от начала файла
time_axis = np.arange(x1.size) / fs + 8000/fs  # начинаем с момента 8000/fs секунд

plt.figure(figsize=[12, 8])

# График 1: как в методичке, но с правильной осью времени
plt.subplot(2, 2, 1)
plt.plot(time_axis, x1, 'b.')  # построение графика цифрового сигнала точками
plt.grid()
plt.xlabel("$t$, c")                      
plt.ylabel("$x[k]$")
plt.title("Цифровой сигнал clarinet.wav\n(диапазон 8000:10000)")

# График 2: тот же диапазон с линиями
plt.subplot(2, 2, 2)
plt.plot(time_axis, x1, 'b-', alpha=0.7)
plt.plot(time_axis, x1, 'r.', markersize=2)
plt.grid()
plt.xlabel("$t$, c")                      
plt.ylabel("$x[k]$")
plt.title("Тот же сигнал с соединенными точками")

# График 3: первые 100 отсчетов файла
plt.subplot(2, 2, 3)
x_start = x_mono[:100]
time_start = np.arange(x_start.size) / fs
plt.stem(time_start, x_start, linefmt='b-', markerfmt='bo', basefmt=' ')
plt.grid()
plt.xlabel("$t$, c")                      
plt.ylabel("$x[k]$")
plt.title("Первые 100 отсчетов от начала файла")

# 4) Определение длины записи
duration = x_mono.size / fs
print(f"\n4) Длина записи:")
print(f"   Количество отсчетов: {x_mono.size}")
print(f"   Длительность: {duration:.2f} секунд")

# 5) Оценка объема файла
print(f"\n5) Оценка объема файла:")

# Теоретический расчет объема
if len(x.shape) > 1:
    num_channels = x.shape[1]
else:
    num_channels = 1

theoretical_size_bits = x.size * bit_depth * num_channels
theoretical_size_bytes = theoretical_size_bits / 8
theoretical_size_kb = theoretical_size_bytes / 1024

print(f"   Теоретический расчет:")
print(f"   - Отсчетов: {x.size}")
print(f"   - Каналов: {num_channels}")
print(f"   - Бит на отсчет: {bit_depth}")
print(f"   - Всего бит: {theoretical_size_bits:,}")
print(f"   - Байт: {theoretical_size_bytes:,.0f}")
print(f"   - Килобайт: {theoretical_size_kb:.2f} КБ")

# Реальный размер файла
actual_size_bytes = os.path.getsize(audio_file)
actual_size_kb = actual_size_bytes / 1024

print(f"\n   Реальный объем файла:")
print(f"   - Байт: {actual_size_bytes:,}")
print(f"   - Килобайт: {actual_size_kb:.2f} КБ")

# Сравнение
print(f"\n   Сравнение:")
print(f"   - Теоретический объем: {theoretical_size_kb:.2f} КБ")
print(f"   - Реальный объем: {actual_size_kb:.2f} КБ")

compression_ratio = actual_size_bytes / theoretical_size_bytes
print(f"   - Отношение (реальный/теоретический): {compression_ratio:.2%}")

if compression_ratio > 1:
    print("   - Реальный файл больше (WAV содержит заголовки)")
else:
    print("   - Реальный файл меньше (возможно сжатие)")

# Дополнительный анализ - спектр сигнала
plt.subplot(2, 2, 4)
from scipy import signal
frequencies, power_spectrum = signal.periodogram(x_mono, fs)
plt.semilogy(frequencies, power_spectrum)
plt.xlabel("Частота, Гц")
plt.ylabel("Спектральная плотность мощности")
plt.title("Спектр сигнала clarinet.wav")
plt.grid(True, alpha=0.3)
plt.xlim(0, 5000)  # ограничиваем до 5 кГц для наглядности

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("ВЫВОДЫ:")
print("=" * 60)
print(f"1) Частота дискретизации: {fs} Гц")
print(f"2) Битная глубина: {bit_depth} бит → {quant_levels} уровней квантования")
print(f"3) Интервал между отсчетами: Δt = {delta_t:.6f} с")
print(f"4) Длительность: {duration:.2f} с")
print(f"5) Теоретический объем: {theoretical_size_kb:.2f} КБ")
print(f"6) Реальный объем: {actual_size_kb:.2f} КБ")
print(f"7) Разница: {abs(theoretical_size_kb - actual_size_kb):.2f} КБ ({compression_ratio:.2%})")