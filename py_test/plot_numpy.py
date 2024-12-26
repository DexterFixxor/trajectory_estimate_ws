import numpy as np
import matplotlib.pyplot as plt

file_name = "trajectory_0000.npy"

data = np.load('./data/data/trajectory_0000.npy')

ids = 50
ide = ids + 50

x = data[ids:ide, 0]
y = data[ids:ide, 1]
z = data[ids:ide, 2]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax.set_xlim([-1000, 1000])
# ax.set_ylim([-1000, 1000])
# ax.set_zlim([-1000, 1000])

ax.scatter(x,y,z)

plt.show()

# np.save(f"./cropped_data/{file_name}, data[ids:ide])
