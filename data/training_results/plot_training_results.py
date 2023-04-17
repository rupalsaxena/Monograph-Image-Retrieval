import numpy as np
import matplotlib.pyplot as plt
import pdb

# pdb.set_trace()
file = 'hypersim.txt'
data = np.loadtxt(file)
train_loss = data[:,2]
train_accuracy = data[:,3]
test_loss = data[:,4]
test_accuracy = data[:,5]
epochs = data[:,0]

# plt.plot(epochs, training, label='training')
# plt.plot(epochs, testing, label='testing')
# plt.title('Performance on the 3DSSG Data Set')
# plt.show()

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Triplet Loss', color=color)
ax1.plot(epochs, train_loss, color=color, label='training loss')
ax1.plot(epochs, test_loss, '-.', color=color, label='testing loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
ax2.plot(epochs, train_accuracy, color=color, label='training accuracy')
ax2.plot(epochs, test_accuracy, '-.', color=color, label='testing accuracy')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
ax2.legend(loc='lower left')
plt.show()