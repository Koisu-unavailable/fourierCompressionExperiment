import numpy as np
from math import pi, sin 
import main

x_values = np.linspace(0, 2*pi, 10**6 )
y_values = [sin(x) for x in x_values]

chunk = main.Period_Chuck_Fn(2*pi, y_values)
print(chunk(1))

