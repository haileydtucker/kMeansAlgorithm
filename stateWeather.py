from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('state_weather.csv')

df.head()

data = df[['temp', 'snowfall']].values

num_states = data.shape[0]
state_names = df['state'].values
state_names

# ---------------------------------------------------------------------

# How many clusters to find
k = 4
print('Running Kmeans on list of', num_states, 'states...')
clt = KMeans(n_clusters=k)
clt.fit(data)
print('Kmeans complete')
# Assign each location to the closest cluster center
labels = clt.predict(data)
print(labels.shape)
labels[:10]

# ---------------------------------------------------------------------

# Visualization
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
plt.figure(figsize=(16, 8))
# Plot states one at a time
for i in range(num_states):
    curr_label = labels[i]
    curr_color = colors[curr_label]
    plt.scatter(data[i, 0], data[i, 1], c=curr_color, s=50)
    plt.text(data[i, 0]+0.5, data[i, 1], state_names[i], size=10)
plt.title('State Weather Data, K = ' + str(k))
plt.xlabel('Average Winter Temp. (F)')
plt.ylabel('Average Snowfall (in)')
plt.show()

