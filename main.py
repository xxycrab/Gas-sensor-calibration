
from TDNN import *
import matplotlib.pyplot as plt
from Linear import *

y1 = linear()
y2 = tdnn()
feature, target = data_read()
target = np.array(target)  # change into ndarray
target = np.reshape(target, (len(target),))  # change into (len,) instead of (len, 1)
test_target = target[2000:3000]
TDL = 12
y3 = test_target[TDL-1:]
y1 = y1[TDL-1:]

x = range(len(test_target)-TDL+1)
#plt.plot(x, y1, color='red', lw=0.85,label='linear', alpha=0.75)
plt.plot(x, y2, color='green', lw=0.85, label='tdnn', alpha=0.75)
plt.plot(x, y3, color='black', lw=0.6, label='golden', linestyle='--', alpha=0.5)
plt.legend()
plt.show()