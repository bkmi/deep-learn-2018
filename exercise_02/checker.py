import numpy as np
import matplotlib.pyplot as plt

with np.load('prediction-challenge-01-data.npz') as fh:
    data_x = fh['data_x']
    data_y = fh['data_y']
    test_x = fh['test_x']

answers_y = np.load('prediction.npy')

ind = np.random.randint(0, test_x.shape[0] - 1)

plt.imshow(test_x[ind, 0])
plt.title(answers_y[ind])
plt.show()


ok = np.loadtxt('prediction.txt')
d = answers_y == ok
print(np.mean(d))
