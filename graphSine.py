import math
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.integrate
from scipy.io import wavfile

import audio

samplerate, data = wavfile.read('./input/60hzSine.wav')
length = data.shape[0] / samplerate
def sin_wave(x):
    return audio.get_amplitude_at_time(x, samplerate, data)

x_values = np.linspace(0, length-1, 100)
sin_values = []
for i in x_values:
    sin_values.append(sin_wave(i))

sin_values = np.array(sin_values)
plt.plot(x_values, sin_values, label='sin(x)', color='red')
plt.title('Graph of 60hz sine wave')
plt.xlabel('x') 
plt.ylabel('f(x)')


plt.show()