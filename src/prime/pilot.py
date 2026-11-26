"""Функции пилотов с семинаров С7 и С8."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def n_per_group(sigma, mde, alpha=0.05, power=0.8):
    """Сколько клиентов нужно на группу, чтобы заметить эффект mde (двусторонний тест)."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return math.ceil(2 * ((z_a + z_b) * sigma / mde) ** 2)


def mde_for_n(sigma, n, alpha=0.05, power=0.8):
    """Какой эффект заметит эксперимент с n клиентами на группу."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return (z_a + z_b) * sigma * math.sqrt(2 / n)


def diff_ci(t, c, alpha=0.05):
    """Разность средних t - c, её (1 - alpha)-интервал и p-значение (Уэлч)."""
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    diff = t.mean() - c.mean()
    v_t, v_c = t.var(ddof=1) / len(t), c.var(ddof=1) / len(c)
    se = math.sqrt(v_t + v_c)
    df = (v_t + v_c) ** 2 / (v_t ** 2 / (len(t) - 1) + v_c ** 2 / (len(c) - 1))
    q = stats.t.ppf(1 - alpha / 2, df)
    p = 2 * stats.t.sf(abs(diff) / se, df)
    return {"diff": diff, "lo": diff - q * se, "hi": diff + q * se, "p": p}


def srm_pvalue(sizes):
    """p-значение проверки SRM: размеры групп против равного деления."""
    return float(stats.chisquare(sizes).pvalue)


def cuped(y, x):
    """Метрика после CUPED и θ: y - θ·(x - mean(x)), θ = cov(x, y) / var(x)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    theta = np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1)
    return y - theta * (x - x.mean()), theta
