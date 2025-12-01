import numpy as np
import matplotlib.pyplot as plt

# Параметры для варианта 1
E = 0.1  # В
tau = 100e-6  # 100 мкс

# Функции для прямоугольных импульсов
def rect_symmetric(t, tau, E):
    """Импульс, симметричный относительно 0 (начинается в -τ/2)"""
    return np.where((t >= -tau/2) & (t <= tau/2), E, 0)

def rect_asymmetric(t, tau, E):
    """Импульс, начинающийся в 0"""
    return np.where((t >= 0) & (t <= tau), E, 0)

# Функция для численного вычисления преобразования Фурье
def fourier_transform(signal_func, f, t, dt, tau, E):
    """Вычисление преобразования Фурье численным интегрированием"""
    X_f = np.zeros(len(f), dtype=complex)
    for i, freq in enumerate(f):
        integrand = signal_func(t, tau, E) * np.exp(-1j * 2 * np.pi * freq * t)
        X_f[i] = np.trapz(integrand, dx=dt)
    return X_f

# Создание временной сетки
t_min, t_max = -2*tau, 2*tau
N_t = 1024
t = np.linspace(t_min, t_max, N_t)
dt = t[1] - t[0]

# Создание частотной сетки
f_min, f_max = -4/tau, 4/tau
N_f = 1000
f = np.linspace(f_min, f_max, N_f)

# Вычисление преобразований Фурье для двух случаев
X_sym = fourier_transform(rect_symmetric, f, t, dt, tau, E)
X_asym = fourier_transform(rect_asymmetric, f, t, dt, tau, E)

# Построение графиков сигналов
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(t*1e6, rect_symmetric(t, tau, E))
plt.title('Импульс, начинающийся в -τ/2 (симметричный)')
plt.xlabel('Время t, мкс')
plt.ylabel('x(t), В')
plt.grid(True)
plt.xlim([t_min*1e6, t_max*1e6])

plt.subplot(1, 2, 2)
plt.plot(t*1e6, rect_asymmetric(t, tau, E))
plt.title('Импульс, начинающийся в 0')
plt.xlabel('Время t, мкс')
plt.ylabel('x(t), В')
plt.grid(True)
plt.xlim([t_min*1e6, t_max*1e6])

plt.tight_layout()
plt.show()

# Построение графиков спектров для симметричного импульса
plt.figure(figsize=(12, 8))

# Симметричный импульс
plt.subplot(2, 3, 1)
plt.plot(f/1e3, np.abs(X_sym)*1e6)
plt.title('|X(f)| для симметричного импульса')
plt.xlabel('Частота f, кГц')
plt.ylabel('|X(f)|, мкВ/Гц')
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(f/1e3, np.real(X_sym)*1e6)
plt.title('Re[X(f)] для симметричного импульса')
plt.xlabel('Частота f, кГц')
plt.ylabel('Re[X(f)], мкВ/Гц')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(f/1e3, np.imag(X_sym)*1e6)
plt.title('Im[X(f)] для симметричного импульса')
plt.xlabel('Частота f, кГц')
plt.ylabel('Im[X(f)], мкВ/Гц')
plt.grid(True)

# Несимметричный импульс
plt.subplot(2, 3, 4)
plt.plot(f/1e3, np.abs(X_asym)*1e6)
plt.title('|X(f)| для импульса с началом в 0')
plt.xlabel('Частота f, кГц')
plt.ylabel('|X(f)|, мкВ/Гц')
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(f/1e3, np.real(X_asym)*1e6)
plt.title('Re[X(f)] для импульса с началом в 0')
plt.xlabel('Частота f, кГц')
plt.ylabel('Re[X(f)], мкВ/Гц')
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(f/1e3, np.imag(X_asym)*1e6)
plt.title('Im[X(f)] для импульса с началом в 0')
plt.xlabel('Частота f, кГц')
plt.ylabel('Im[X(f)], мкВ/Гц')
plt.grid(True)

plt.tight_layout()
plt.show()

# Сравнение модулей спектров
plt.figure(figsize=(10, 6))
plt.plot(f/1e3, np.abs(X_sym)*1e6, label='Симметричный (начало в -τ/2)', linewidth=2)
plt.plot(f/1e3, np.abs(X_asym)*1e6, '--', label='Несимметричный (начало в 0)', linewidth=2)
plt.title('Сравнение модулей спектров')
plt.xlabel('Частота f, кГц')
plt.ylabel('|X(f)|, мкВ/Гц')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Проверка теоремы запаздывания
# Теоретически: X₂(f) = X₁(f) * exp(-j*2π*f*t₀), где t₀ = τ/2 (сдвиг от симметричного к несимметричному)
t0 = tau/2  # сдвиг времени
X_theoretical = X_sym * np.exp(-1j * 2 * np.pi * f * t0)

# Сравнение вычисленного и теоретического спектров для несимметричного импульса
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(f/1e3, np.abs(X_asym)*1e6, label='Вычисленный')
plt.plot(f/1e3, np.abs(X_theoretical)*1e6, '--', label='Теоретический (с учетом запаздывания)')
plt.title('Сравнение |X(f)|')
plt.xlabel('Частота f, кГц')
plt.ylabel('|X(f)|, мкВ/Гц')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(f/1e3, np.real(X_asym)*1e6, label='Вычисленный')
plt.plot(f/1e3, np.real(X_theoretical)*1e6, '--', label='Теоретический')
plt.title('Сравнение Re[X(f)]')
plt.xlabel('Частота f, кГц')
plt.ylabel('Re[X(f)], мкВ/Гц')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(f/1e3, np.imag(X_asym)*1e6, label='Вычисленный')
plt.plot(f/1e3, np.imag(X_theoretical)*1e6, '--', label='Теоретический')
plt.title('Сравнение Im[X(f)]')
plt.xlabel('Частота f, кГц')
plt.ylabel('Im[X(f)], мкВ/Гц')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Аналитические выражения спектральных плотностей
print("Аналитические формулы:")
print("1. Для импульса, симметричного относительно 0 (начало в -τ/2):")
print("   X₁(f) = E * τ * sinc(πfτ)")
print("   где sinc(x) = sin(x)/x")
print()
print("2. Для импульса, начинающегося в 0:")
print("   X₂(f) = X₁(f) * exp(-jπfτ)")
print("   Re[X₂(f)] = E * τ * sinc(πfτ) * cos(πfτ)")
print("   Im[X₂(f)] = -E * τ * sinc(πfτ) * sin(πfτ)")
print()
print("Теорема запаздывания подтверждается:")
print("1. Модули спектров |X₁(f)| и |X₂(f)| совпадают")
print("2. Разница в фазе: arg[X₂(f)] = arg[X₁(f)] - πfτ")
print(f"3. Сдвиг во времени: t₀ = τ/2 = {tau/2*1e6:.1f} мкс")