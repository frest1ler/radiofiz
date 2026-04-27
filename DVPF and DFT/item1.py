import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Параметры варианта
N = 8
L = 4
nu_0 = 0.1

# Частотная ось
nu = np.linspace(-0.5, 0.5, 1000)

# ============================================
# ЗАДАЧА 1.1: Прямоугольный импульс
# ============================================
x_N = np.ones(N)

def DFT_manual(x, nu_vec):
    """Ручное вычисление ДВПФ"""
    N_len = len(x)
    X = np.zeros(len(nu_vec), dtype=complex)
    for i, nu_val in enumerate(nu_vec):
        X[i] = np.sum(x * np.exp(-1j * 2 * np.pi * nu_val * np.arange(N_len)))
    return X

X_N = DFT_manual(x_N, nu)

# Аналитическая формула (для сравнения) - ИСПРАВЛЕННАЯ ВЕРСИЯ
def X_N_analytical(nu):
    """Аналитическое выражение для ДВПФ прямоугольного импульса"""
    # Преобразуем в массив, если передано скалярное значение
    nu = np.asarray(nu)
    result = np.zeros_like(nu, dtype=complex)
    
    # Для точек, где знаменатель близок к нулю (особые точки)
    mask_small = np.abs(np.sin(np.pi * nu)) < 1e-10
    mask_not_small = ~mask_small
    
    # В особых точках (nu = 0, ±1, ±2, ...)
    result[mask_small] = N
    
    # В остальных точках
    if np.any(mask_not_small):
        nu_vals = nu[mask_not_small]
        result[mask_not_small] = (np.sin(np.pi * N * nu_vals) / 
                                 np.sin(np.pi * nu_vals) * 
                                 np.exp(-1j * np.pi * nu_vals * (N - 1)))
    
    return result

X_N_anal = X_N_analytical(nu)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Графики модуля
axes[0, 0].plot(nu, np.abs(X_N), label='Моделирование', linewidth=2)
axes[0, 0].plot(nu, np.abs(X_N_anal), '--', label='Аналитическое', alpha=0.7)
axes[0, 0].set_xlabel('ν')
axes[0, 0].set_ylabel('|X_N(ν)|')
axes[0, 0].set_title('Задача 1.1: Модуль ДВПФ')
axes[0, 0].grid(True)
axes[0, 0].legend()

# Графики фазы
axes[0, 1].plot(nu, np.angle(X_N), label='Моделирование', linewidth=2)
axes[0, 1].plot(nu, np.angle(X_N_anal), '--', label='Аналитическое', alpha=0.7)
axes[0, 1].set_xlabel('ν')
axes[0, 1].set_ylabel('arg(X_N(ν))')
axes[0, 1].set_title('Задача 1.1: Фаза ДВПФ')
axes[0, 1].grid(True)
axes[0, 1].legend()

# ============================================
# ЗАДАЧА 1.2: Свойство масштабирования
# ============================================
x_L = np.zeros(N * L)
x_L[::L] = x_N
X_L = DFT_manual(x_L, nu)

axes[1, 0].plot(nu, np.abs(X_L), label='|X_L(ν)|', linewidth=2)
axes[1, 0].plot(nu, np.abs(X_N_analytical(nu * L)), '--', label='|X_N(νL)|', alpha=0.7)
axes[1, 0].set_xlabel('ν')
axes[1, 0].set_ylabel('|X(ν)|')
axes[1, 0].set_title('Задача 1.2: Свойство масштабирования')
axes[1, 0].grid(True)
axes[1, 0].legend()

# ============================================
# ЗАДАЧА 1.3: Дифференцирование спектральной плотности
# ============================================
k = np.arange(N)
x_D = k * x_N
X_D = DFT_manual(x_D, nu)

# Численное дифференцирование X_N
dX_N = np.gradient(X_N_anal, nu)  # производная аналитического выражения
X_diff = (1j/(2*np.pi)) * dX_N

axes[1, 1].plot(nu, np.abs(X_D), label='|X_D(ν)|', linewidth=2)
axes[1, 1].plot(nu, np.abs(X_diff), '--', label='|(j/2π)dX_N/dν|', alpha=0.7)
axes[1, 1].set_xlabel('ν')
axes[1, 1].set_ylabel('|X(ν)|')
axes[1, 1].set_title('Задача 1.3: Дифференцирование спектра')
axes[1, 1].grid(True)
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# ============================================
# ЗАДАЧА 1.4: Теорема смещения
# ============================================
k = np.arange(N)
x_S = x_N * np.exp(1j * 2 * np.pi * nu_0 * k)
X_S = DFT_manual(x_S, nu)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(nu, np.abs(X_S), linewidth=2)
ax[0].set_xlabel('ν')
ax[0].set_ylabel('|X_S(ν)|')
ax[0].set_title('Задача 1.4: Модуль ДВПФ (смещение)')
ax[0].grid(True)

# Для наглядности покажем также |X_N(ν - ν_0)|
ax[1].plot(nu, np.abs(X_N_analytical(nu - nu_0)), linewidth=2)
ax[1].set_xlabel('ν')
ax[1].set_ylabel('|X_N(ν - ν₀)|')
ax[1].set_title('Задача 1.4: Аналитическое выражение')
ax[1].grid(True)

plt.tight_layout()
plt.show()

# ============================================
# ЗАДАЧА 1.5: Теорема о свертке
# ============================================
# Линейная свертка
conv_result = signal.convolve(x_N, x_N, mode='full')
M = len(conv_result)
X_conv = DFT_manual(conv_result, nu)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].stem(np.arange(M), conv_result, linefmt='C0-', markerfmt='C0o', basefmt='C0-')
ax[0].set_xlabel('k')
ax[0].set_ylabel('x_N * x_N')
ax[0].set_title('Задача 1.5: Свертка (временная область)')
ax[0].grid(True)

ax[1].plot(nu, np.abs(X_conv), label='Моделирование', linewidth=2)
ax[1].plot(nu, np.abs(X_N_analytical(nu))**2, '--', label='|X_N(ν)|²', alpha=0.7)
ax[1].set_xlabel('ν')
ax[1].set_ylabel('|X(ν)|')
ax[1].set_title('Задача 1.5: Модуль ДВПФ свертки')
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.show()

# ============================================
# ВЫЧИСЛЕНИЕ ТАБЛИЧНЫХ ЗНАЧЕНИЙ
# ============================================
print("="*60)
print("ТАБЛИЧНЫЕ ЗНАЧЕНИЯ ДЛЯ ЗАДАЧИ 1.1:")
print("="*60)

# X(0)
X_0 = N
print(f"X(0) = {X_0}")

# Ширина главного лепестка на нулевом уровне
# Для прямоугольного импульса из N отсчетов: Δν = 2/N
delta_nu = 2 / N
print(f"Ширина главного лепестка Δν = {delta_nu:.3f}")

# Точки скачков фазы на π
# Фаза меняется на π при ν = m/N, где m - целое, кроме кратных N
m_vals = np.arange(-N//2 + 1, N//2)
phase_jump_points = m_vals / N
print(f"Точки скачков фазы на π: {phase_jump_points[phase_jump_points != 0]}")

# Энергия (по теореме Парсеваля)
energy = np.sum(np.abs(x_N)**2)  # = N для единичных импульсов
print(f"Энергия (по Парсевалю) = {energy}")

print("\n" + "="*60)
print("ТАБЛИЧНЫЕ ЗНАЧЕНИЯ ДЛЯ ЗАДАЧИ 1.5:")
print("="*60)

# X(0) для свертки
X_conv_0 = np.sum(conv_result)
print(f"X(0) = {X_conv_0}")

# Ширина главного лепестка (остается такой же)
print(f"Ширина главного лепестка Δν = {delta_nu:.3f}")

# Энергия свертки
energy_conv = np.sum(np.abs(conv_result)**2)
print(f"Энергия (по Парсевалю) = {energy_conv}")