import json
import numpy as np
import seaborn as sns
import pylab as plt

with open("events.json", "r") as f:
    events = json.load(f)

# start when plan started
filtered = {}
start_time = min(events["Plan was created"])
for k, v in events.items():
    for _v in v:
        if _v > start_time:
            filtered.setdefault(k, []).append(_v)
events = filtered

# init matrix
num_events = len(list(events.keys()))
num_bins = 100
_alltimestamps = list(flatten(v for v in events.values()))
mn = min(_alltimestamps)
mx = max(_alltimestamps)
bins = np.linspace(mn, mx, num_bins)
X = np.zeros((num_events, num_bins))

# key unique events
event_keys = sorted(list(events.keys()))

for i, k in enumerate(event_keys):
    print(k)
    for _x in np.digitize(events[k], bins):
        X[i][_x - 1] += 1

# Plot Figure
xfiltered = X[:, :]
fig = plt.figure(figsize=(10, 3))
ax = fig.gca()
sns.heatmap(
    xfiltered,
    ax=ax,
    square=False,
    linewidths=0.1,
    mask=np.where(xfiltered == 0, True, False),
    yticklabels=event_keys,
)
plt.savefig("planevents.svg", format="svg")
