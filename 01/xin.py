import matplotlib.pyplot as plt
import numpy as np

# 创建一个6x6的网格
grid = np.zeros((8, 9))

# 定义心形的像素位置
heart_shape = [
    (1, 2), (1, 6),
    (2, 1), (2, 2), (2, 3), (2, 5), (2, 6), (2, 7),
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7),
    (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
    (5, 3), (5, 4), (5, 5),
    (6, 4)
]

# 设置心形区域为1
for pos in heart_shape:
    grid[pos] = 1

# 绘制网格
fig, ax = plt.subplots()
ax.imshow(grid, cmap="Blues", origin="upper")
# 设置心形区域为橙色
for pos in heart_shape:
    ax.add_patch(plt.Rectangle((pos[1]-0.5, pos[0]-0.5), 1, 1, fill=True, color="orange", edgecolor="white"))
# 去除坐标轴
ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
plt.show()
