import numpy as np

name_list = ['60', '64', '67', '72', '76']
params = [np.load('../results/isolated_sounds/params_' + name_list[i] + '_1-down.npz')
          for i in range(0, 5)]
