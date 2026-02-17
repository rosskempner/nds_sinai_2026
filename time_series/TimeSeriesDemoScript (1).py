#%%


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# Load data
data = np.loadtxt('LFP.txt')
events = np.loadtxt('events.txt', dtype=int)

# Visualize the time series with events
plt.figure()
plt.plot(data, color=[0.5, 0.5, 0.5])
for event in events:
    plt.axvline(x=event, color='r', linewidth=0.5)
plt.xlabel('Time (ms)')
plt.ylabel('LFP')
plt.tight_layout()
plt.gca().tick_params(labelsize=16)
plt.show(block=False)

#%%
# Separate the time series into "trials"
pre, post = 1000, 2000
times = np.arange(-pre, post + 1)
trials = np.array([data[event - pre:event + post + 1] for event in events])

# Visualize a few trials stacked up
nTr = 7
yax = 0.5 * np.arange(1, nTr + 1)
plt.figure()
for i in range(nTr):
    plt.plot(times, yax[i] + trials[i], color=[0.5, 0.5, 0.5])
plt.axvline(x=0, color='r', linewidth=0.5)
plt.yticks(yax, labels=np.arange(1, nTr + 1))
plt.xlabel('Time from event (ms)')
plt.ylabel('Trials')
plt.gca().tick_params(labelsize=16)
#plt.show()

# Visualize all trials in a heatmap
plt.figure()
plt.imshow(trials, aspect='auto', extent=[times[0], times[-1], 0, trials.shape[0]], cmap='hot', origin='lower')
plt.axvline(x=0, color='r', linewidth=0.5)
plt.xlabel('Time from event (ms)', fontsize=16)
plt.ylabel('Trials', fontsize=16)
plt.colorbar()
plt.gca().tick_params(labelsize=16)
#plt.show()

# Visualize the average response across trials
plt.figure()
line_raw, = plt.plot(times, trials.mean(axis=0), color=[0.5, 0.5, 0.5], label='Raw')
line_event = plt.axvline(x=0, color='r', linewidth=0.5)  # No label for event line
plt.xlabel('Time from event (ms)')
plt.ylabel('Average event-related response')
plt.gca().tick_params(labelsize=16)
#plt.show()

# Smooth the original data using moving average
window_size = 100
data_sm = np.convolve(data, np.ones(window_size)/window_size, mode='same')
trials_sm = np.array([data_sm[event - pre:event + post + 1] for event in events])
line_ma, = plt.plot(times, trials_sm.mean(axis=0), color='k', linewidth=2, label='Moving Average')

# Smooth using Gaussian filter
data_sm_gauss = gaussian_filter1d(data, sigma=100)
trials_sm_gauss = np.array([data_sm_gauss[event - pre:event + post + 1] for event in events])
line_gauss, = plt.plot(times, trials_sm_gauss.mean(axis=0), color=[0, 0.8, 0.3], linewidth=2, label='Gaussian Smooth')

plt.legend(handles=[line_raw, line_ma, line_gauss])
plt.show()
