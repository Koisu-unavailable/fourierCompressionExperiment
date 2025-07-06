import math
from math import pi, floor

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.integrate
from scipy.io import wavfile

import audio
import fourier
import chunk
import time

samplerate, data = wavfile.read("./input/BadApple.wav")
data = data[:, 0]
length = data.shape[0] / samplerate

chunks = chunk.chunk(data, 2)
current_chunk = chunks[4]
current_chunk_length = current_chunk.shape[0] / samplerate


def chunk_func(x):
    return audio.get_amplitude_at_time(x, samplerate, current_chunk)


class Period_Chuck_Fn:
    def __init__(self, period_length: int, data: np.ndarray):
        self.period_length = period_length
        self.data = data

    def periodic_chunk_fn(self, x):
        there = False
        act_value = x
        times_subtracted = 0
        if act_value <= self.period_length:
            return self.data[floor(x)]
        while not there:
            if act_value <= 0 or act_value - self.period_length <= 0:
                act_value = 0
                break
            act_value -= self.period_length

            there = act_value <= self.period_length

            times_subtracted += 1
        return self.data[floor(act_value)]

    def __call__(self, *args, **kwds):
        return self.periodic_chunk_fn(x=kwds["x"])


def complete(x):
    return audio.get_amplitude_at_time(x, samplerate, data)




# x_values = np.linspace(0, length-1, 100)
# full_func = []
# chunk_y_values = []
# fourier_chunk = []
# for i in x_values:
#     try:
#         full_func.append(complete(i))
#     except IndexError:
#         continue
# x_values_chunk = np.linspace(0,(chunks[4].shape[0] / samplerate), 100)
# for x in x_values_chunk:
#     try:
#         fourier_chunk.append(fourier_expansion(x, 100))
#         chunk_y_values.append(chunk_func(x))
#     except IndexError:
#         continue
chunk_fn = Period_Chuck_Fn(len(current_chunk), current_chunk)
fourier_expansion = fourier.get_fourier_expansion_func(chunk_fn.periodic_chunk_fn)
x_values = np.linspace(0, chunk_fn.period_length, 100, endpoint=False)
x_values_to_remove = []
y_values = []
y2_values = []
previous_times = []
average = 0
for x in x_values:
    try:
        y_values.append(chunk_fn(x=x))
        start = time.perf_counter()
        y2_values.append(fourier_expansion(x, 10))
        end = time.perf_counter()
        elasped = end - start
        print(elasped)
        previous_times.append(elasped)
        average = np.average(previous_times)
        print(f"Average: {average}")
        amount_left = x_values.shape[0] - list(x_values).index(x)
        print(f"Estimated minutes left: {((average * amount_left)/60) if average * amount_left > 1 else (average * amount_left)}")
        
    except IndexError as e:
        x_values_to_remove.append(x)

for x in x_values_to_remove:
    x_values = np.delete(x_values, floor(list(x_values).index(x)))


# full_func = np.array(full_func)
# chunk_y_values = np.array(chunk_y_values)
# fourier_chunk = np.array(fourier_chunk)
# print(type(chunk_y_values[0]))
plt.plot(x_values, y_values, label="Periodic ver. of chunk", color="red")
plt.plot(x_values, y2_values, label="Fourier expanded", color="purple")
plt.legend()
plt.title("Periodic verison of chunk")
plt.xlabel("x")
plt.ylabel("f(x)")


plt.show()
