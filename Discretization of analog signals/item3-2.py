import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import os
import sys

# Параметры для варианта 1
K1 = 35  # коэффициент прореживания 1
K2 = 70  # коэффициент прореживания 2

# Проверяем наличие файла в нескольких возможных местах
file_paths = [
    'tuning-fork.wav',
    'audio_files/tuning-fork.wav',
    './audio_files/tuning-fork.wav'
]

file_found = None
for path in file_paths:
    if os.path.exists(path):
        file_found = path
        break

if file_found is None:
    print("ОШИБКА: Файл tuning-fork.wav не найден!")
    print("Убедитесь, что файл находится в одной из следующих локаций:")
    for path in file_paths:
        print(f"  - {path}")
    sys.exit(1)  # Останавливаем программу

# Загрузка файла
try:
    fs, x = wav.read(file_found)
    print(f"Файл успешно загружен: {file_found}")
    print(f"Частота дискретизации: {fs} Гц")
except Exception as e:
    print(f"Ошибка при чтении файла: {e}")
    sys.exit(1)

# Если файл стерео, берем один канал
if len(x.shape) > 1:
    print("Файл стерео, берем первый канал")
    x = x[:, 0]

# Функция для вычисления ДВПФ
def DTFT_abs(xk, fs, M=2**17):
    """Вычисление модуля ДВПФ"""
    res = np.abs(np.fft.fftshift(np.fft.fft(xk, M)))
    freq = np.fft.fftshift(np.fft.fftfreq(M, 1/fs))
    return freq, res

# 1. Спектр исходного сигнала
plt.figure(figsize=[8, 3], dpi=100)
freq_orig, spec_orig = DTFT_abs(x/fs, fs)
plt.plot(freq_orig, spec_orig)
plt.grid()
plt.title("ДВПФ исходного сигнала")
plt.xlabel("Частота $f$, Гц")
plt.ylabel("$|X(f)|$")
plt.xlim([-fs/2, fs/2])
plt.tight_layout()
plt.show()

# 2. Прореживание с коэффициентом K1 (без фильтрации)
plt.figure(figsize=[8, 3], dpi=100)
y1 = x[::K1]  # прореживание - берем каждый K1-й отсчет
fs1 = fs / K1  # новая частота дискретизации после прореживания
freq1, spec1 = DTFT_abs(y1/fs1, fs1)
plt.plot(freq1, spec1)
plt.grid()
plt.title(f"ДВПФ прореженного сигнала $K_1=${K1}")
plt.xlabel("Частота $f$, Гц")
plt.ylabel("$|X(f)|$")
plt.xlim([-fs1/2, fs1/2])
plt.tight_layout()
plt.show()

# 3. Прореживание с коэффициентом K2 (без фильтрации)
plt.figure(figsize=[8, 3], dpi=100)
y2 = x[::K2]  # прореживание - берем каждый K2-й отсчет
fs2 = fs / K2  # новая частота дискретизации после прореживания
freq2, spec2 = DTFT_abs(y2/fs2, fs2)
plt.plot(freq2, spec2)
plt.grid()
plt.title(f"ДВПФ прореженного сигнала $K_2=${K2}")
plt.xlabel("Частота $f$, Гц")
plt.ylabel("$|X(f)|$")
plt.xlim([-fs2/2, fs2/2])
plt.tight_layout()
plt.show()

# Анализ результатов
print("\n" + "=" * 60)
print("АНАЛИЗ РЕЗУЛЬТАТОВ ДЛЯ ВАРИАНТА 1")
print("=" * 60)
print(f"Исходный сигнал: частота дискретизации fs = {fs} Гц")
print(f"Коэффициент прореживания K1 = {K1}")
print(f"Коэффициент прореживания K2 = {K2}")
print()

# Характеристики после прореживания
print("1. Характеристики после прореживания:")
print(f"   а) Для K1 = {K1}:")
print(f"      - Новая частота дискретизации: fs1 = fs/K1 = {fs1:.0f} Гц")
print(f"      - Частота Найквиста: f_nyquist1 = fs1/2 = {fs1/2:.0f} Гц")
print()

print(f"   б) Для K2 = {K2}:")
print(f"      - Новая частота дискретизации: fs2 = fs/K2 = {fs2:.0f} Гц")
print(f"      - Частота Найквиста: f_nyquist2 = fs2/2 = {fs2/2:.0f} Гц")
print()

print("2. Принципиальное отличие между спектрами:")
print("   - При прореживании уменьшается частота дискретизации")
print("   - Уменьшается частота Найквиста (максимальная частота, которую можно представить)")
print("   - Спектральные компоненты выше новой частоты Найквиста 'отражаются' обратно")
print("   - Это создает ложные частотные компоненты (эффект наложения или алиасинг)")
print()

print("3. Обусловлено эффектами:")
print("   - Нарушение условия Найквиста при прореживании")
print("   - Отсутствие антиалиасингового фильтра перед прореживанием")
print("   - Отражание высокочастотных компонент в низкочастотную область")
print("=" * 60)