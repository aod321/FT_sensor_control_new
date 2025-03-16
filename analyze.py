import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.signal as sig

history = pickle.load(open("data.pkl", "rb"))

z = history.T

# apply low pass filter
b, a = sig.butter(4, 0.2)
z_filtered = sig.filtfilt(b, a, z)

print(z_filtered.T[-1])

plt.plot(z[3], label="raw")
plt.plot(z_filtered[3], label="filtered")

plt.show()