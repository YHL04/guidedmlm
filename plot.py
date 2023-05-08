import pandas as pd
import os
import matplotlib.pyplot as plt

dir = os.listdir("logs/")
dir.sort()
file1 = "logs/" + dir[-1]
file2 = "logs/" + dir[-2]


data1 = pd.read_csv(file1,
                    names=["Loss", "Discriminator_Loss"]
                    )
data2 = pd.read_csv(file2,
                    names=["Loss", "Discriminator_Loss"]
                    )

plt.subplot(3, 1, 1)
plt.yscale("log")
plt.plot(data1["Loss"])
plt.plot(data2["Loss"])

# plt.subplot(3, 1, 2)
# plt.yscale("log")
# plt.plot(data["Generator_Loss"])

plt.subplot(3, 1, 3)
plt.yscale("log")
plt.plot(data1["Discriminator_Loss"])
plt.plot(data2["Discriminator_Loss"])

plt.show()
