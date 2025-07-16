from math import pi, cos, sin
from typing import Callable
import scipy.integrate
def get_fourier_expansion_func(wave: Callable[[float], float]) -> Callable[[float, int], float]:
    period = pi * 2

    a_0 = (1/period) * scipy.integrate.quad(wave, 0, period, limit=2000)[0]
    f = 1/period
    little_omega = 2 * pi * f
    def a(n):
        return (2/period) * scipy.integrate.quad(lambda t: wave(t) * cos(n*little_omega*t), 0, period, limit=2000)[0]
    def b(n):
        return (2/period) * scipy.integrate.quad(lambda t: wave(t) * sin(n*little_omega*t), 0, period, limit=2000)[0]

    def func_to_sum(n, variable):
        return (a(n) * cos(n*little_omega*variable)) + (b(n) * sin(n*little_omega*variable))


    def expanded_fourier_series(variable, precision):
        result = a_0
        for i in range(1, precision):
            result += func_to_sum(i, variable)
        return result
    return expanded_fourier_series