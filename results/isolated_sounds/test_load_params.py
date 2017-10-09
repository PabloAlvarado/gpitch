import numpy as np
import matplotlib.pyplot as plt

loaded = np.load('params_comp_16_khz_MAPS_ISOL_NO_P_S0_M62_AkPnBcht.npz')

plt.figure()
plt.plot(loaded['x'], loaded['y'])