import numpy as np
import matplotlib.pyplot as plt

# Параметры для варианта 1
tau = 100e-6  # 100 мкс
fs = 10 / tau  # частота дискретизации

# Функция прямоугольного импульса длительностью 1.05τ
def boxcar(t, tau):
    if 0 <= t <= 1.05 * tau:
        return 1.0
    else:
        return 0.0

# Функция для вычисления DTFT
def DTFT_abs3(xk, fs, M=2048):
    """Вычисление модуля ДВПФ"""
    res = np.abs(np.fft.fftshift(np.fft.fft(xk, M)))
    # Расширяем для отображения нескольких периодов
    freq = np.fft.fftshift(np.fft.fftfreq(M, 1/fs))
    return freq, res

# Функция sinc-интерполяции (ряд Котельникова)
def sinc_phi(t, k, fs):
    dt = 1/fs
    if np.isclose(0, t - k * dt):
        return 1.0
    else:
        return np.sin(np.pi * fs * (t - k * dt)) / (np.pi * fs * (t - k * dt))

def sinc_interp(t, xk, fs):
    """Восстановление сигнала по теореме Котельникова"""
    result = np.zeros_like(t)
    for i, t_val in enumerate(t):
        sum_val = 0
        for k in range(len(xk)):
            sum_val += xk[k] * fs * sinc_phi(t_val, k, fs)
        result[i] = sum_val
    return result

# Временная сетка для аналогового сигнала
t_band = np.linspace(-0.2*tau, 2*tau, 1024)

# Дискретизация
tk = np.arange(0, 2*tau + 1.0/fs, 1.0/fs)  # моменты дискретизации
xk = np.array([boxcar(tk1, tau) for tk1 in tk])  # отсчеты

# Частотная сетка для спектров
f_band = np.linspace(-1.5*fs, 1.5*fs, 1000)

# Вычисление спектра аналогового сигнала (прямоугольного импульса)
def analog_spectrum(f, tau):
    """Аналитический спектр прямоугольного импульса длительностью 1.05τ"""
    f_safe = np.where(np.abs(f) < 1e-12, 1e-12, f)
    return 1.05 * tau * np.abs(np.sin(np.pi * 1.05 * tau * f_safe) / (np.pi * 1.05 * tau * f_safe))

# Построение графиков
plt.figure(figsize=[10, 8])

# График 1: Сигналы во временной области
plt.subplot(2, 1, 1)
plt.plot(t_band * 1e6, [boxcar(t, tau) for t in t_band], 'g', 
         label='Прямоугольный импульс (1.05τ)', linewidth=2)
plt.stem(tk * 1e6, xk * fs, linefmt='b', markerfmt='bo', basefmt=' ', 
         label=f'Отсчеты (fs={fs/1e3:.1f} кГц)')

# Восстановленный сигнал
t_recon = np.linspace(-0.2*tau, 2*tau, 500)
x_recon = sinc_interp(t_recon, xk, fs)
plt.plot(t_recon * 1e6, x_recon, 'c', 
         label='Восстановленный сигнал', linewidth=1.5, alpha=0.7)

plt.title(f"Сигнал: τ={tau*1e6} мкс, fs={fs/1e3:.1f} кГц")
plt.xlabel("Время t, мкс")
plt.ylabel("x(t), В")
plt.grid(True)
plt.legend(loc='upper right')
plt.xlim([-0.2*tau*1e6, 2*tau*1e6])

# График 2: Спектры
plt.subplot(2, 1, 2)

# Спектр аналогового сигнала
X_analog = analog_spectrum(f_band, tau)
plt.plot(f_band/1e3, X_analog * 1e6, 'g', 
         label='Спектр прямоугольного импульса', linewidth=2)

# Спектр дискретизованного сигнала (DTFT)
freq_dtft, X_dtft = DTFT_abs3(xk, fs, M=2048)
plt.plot(freq_dtft/1e3, X_dtft * 1e6, 'b', 
         label='ДВПФ дискретизованного сигнала', linewidth=1.5)

# Линии частоты Найквиста
f_nyquist = fs / 2
plt.axvline(x=f_nyquist/1e3, color='r', linestyle='--', alpha=0.5, 
            label=f'Частота Найквиста = {f_nyquist/1e3:.1f} кГц')
plt.axvline(x=-f_nyquist/1e3, color='r', linestyle='--', alpha=0.5)

# Области наложения спектров
plt.axvspan(f_nyquist/1e3, 1.5*fs/1e3, alpha=0.1, color='red', 
            label='Область наложения спектров')
plt.axvspan(-1.5*fs/1e3, -f_nyquist/1e3, alpha=0.1, color='red')

plt.title("Спектры: эффект наложения")
plt.xlabel("Частота f, кГц")
plt.ylabel("|X(f)|, мкВ/Гц")
plt.grid(True)
plt.legend(loc='upper right')
plt.xlim([-1.5*fs/1e3, 1.5*fs/1e3])

plt.tight_layout()
plt.show()

# Анализ результатов
print("=" * 60)
print("АНАЛИЗ РЕЗУЛЬТАТОВ")
print("=" * 60)
print(f"Параметры: τ = {tau*1e6} мкс, fs = {fs/1e3:.1f} кГц")
print(f"Частота Найквиста: f_nyquist = {f_nyquist/1e3:.1f} кГц")
print()

# Ширина спектра прямоугольного импульса
bandwidth = 1 / (1.05 * tau)
print(f"1. Характеристики прямоугольного импульса:")
print(f"   - Длительность: 1.05τ = {1.05*tau*1e6:.1f} мкс")
print(f"   - Ширина спектра (по первому нулю): {bandwidth/1e3:.1f} кГц")
print()

# Проверка условия Найквиста
print(f"2. Проверка условия Найквиста:")
print(f"   - Ширина спектра импульса: {bandwidth/1e3:.1f} кГц")
print(f"   - Частота Найквиста: {f_nyquist/1e3:.1f} кГц")
if bandwidth > f_nyquist:
    print(f"   ⚠️  Нарушено условие Найквиста: {bandwidth/1e3:.1f} > {f_nyquist/1e3:.1f} кГц")
    print(f"   → Происходит наложение спектров (алиасинг)")
else:
    print(f"   ✓ Условие Найквиста выполнено")
print()

print(f"3. Наблюдение эффекта наложения:")
print(f"   - На графике спектра видно, что спектры соседних периодов перекрываются")
print(f"   - Это приводит к искажению восстановленного сигнала")
print()

# Сигнал без наложения (удовлетворяющий условию Найквиста)
print(f"4. Сигнал без эффекта наложения:")
print(f"   Сигнал, который восстанавливается точно по теореме Котельникова,")
print(f"   должен удовлетворять условию Найквиста: f_max < {f_nyquist/1e3:.1f} кГц")
print(f"   Пример: синусоида с частотой f0 = {0.8*f_nyquist/1e3:.1f} кГц")
print()

# Пример сигнала без наложения
def signal_no_aliasing(t, fs):
    """Сигнал, удовлетворяющий условию Найквиста"""
    f0 = 0.8 * fs / 2  # частота меньше Найквиста
    return np.sin(2 * np.pi * f0 * t)

# Дополнительный график: сравнение с сигналом без наложения
plt.figure(figsize=[10, 4])

# Сигнал без наложения
t_smooth = np.linspace(0, 2*tau, 1000)
x_smooth = signal_no_aliasing(t_smooth, fs)

# Дискретизация сигнала без наложения
xk_smooth = signal_no_aliasing(tk, fs)
x_recon_smooth = sinc_interp(t_recon, xk_smooth, fs)

plt.subplot(1, 2, 1)
plt.plot(t_smooth * 1e6, x_smooth, 'g', label='Исходный сигнал')
plt.stem(tk * 1e6, xk_smooth * fs, linefmt='b', markerfmt='bo', basefmt=' ', 
         label='Отсчеты')
plt.plot(t_recon * 1e6, x_recon_smooth, 'c', label='Восстановленный')
plt.title(f"Сигнал без наложения\nf0 = {0.8*f_nyquist/1e3:.1f} кГц < f_Nyq")
plt.xlabel("Время t, мкс")
plt.ylabel("x(t), В")
plt.grid(True)
plt.legend()
plt.xlim([0, 2*tau*1e6])

# Спектры
f_band_smooth = np.linspace(-1.5*fs, 1.5*fs, 1000)
X_smooth_spectrum = np.where(np.abs(np.abs(f_band_smooth) - 0.8*f_nyquist) < 100, 1, 0)

plt.subplot(1, 2, 2)
plt.plot(f_band_smooth/1e3, X_smooth_spectrum * 1e6, 'g', 
         label='Спектр сигнала')
plt.axvline(x=f_nyquist/1e3, color='r', linestyle='--', alpha=0.5, 
            label='Частота Найквиста')
plt.axvline(x=-f_nyquist/1e3, color='r', linestyle='--', alpha=0.5)
plt.title("Спектр без наложения")
plt.xlabel("Частота f, кГц")
plt.ylabel("|X(f)|, мкВ/Гц")
plt.grid(True)
plt.legend()
plt.xlim([-1.5*fs/1e3, 1.5*fs/1e3])

plt.tight_layout()
plt.show()

print("5. Вывод:")
print("   - При нарушении условия Найквиста происходит наложение спектров")
print("   - Это приводит к невозможности точного восстановления сигнала")
print("   - Для точного восстановления по теореме Котельникова необходимо,")
print(f"     чтобы максимальная частота сигнала была меньше {f_nyquist/1e3:.1f} кГц")
print("=" * 60)