import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.signal as sig

history = pickle.load(open("data.pkl", "rb"))

# turn deque object into 2d array
history = np.array(history)

z = history.T

# apply low pass filter
b, a = sig.butter(5, 1/15)
z_filtered = sig.filtfilt(b, a, z)

CHANNEL = 1  # the channel to plot (0-5)
plt.plot(z[CHANNEL], label="raw")
plt.plot(z_filtered[CHANNEL], label="filtered")
plt.legend()

plt.show()
