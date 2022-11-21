import numpy as np
import matplotlib.pyplot as plt


def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.figure(figsize=(10, 6))
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14, labels=label)
    plt.grid(True)
    plt.show()


# Trend
def trend(time, slope=0):
    series = slope * time
    return series


time = np.arange(365)

# define the slope
slope = 0.1

# generate measurements with the slope
series = trend(time, slope)

# Plot the results
plot_series(time, series, label=[f'slope={slope}'])

# Seasonality
def seasonal_pattern(season_time):
    data_pattern = np.where(season_time < 0.4,
                            np.cos(season_time * 2 * np.pi),
                            1 / np.exp(3 * season_time))
    return data_pattern

def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    print("season_time: ", season_time)
    print("season_time[:10]: ", season_time[:10])
    data_pattern = amplitude * seasonal_pattern(season_time)
    return data_pattern


time = np.arange(4 * 365 + 1)
period = 365
amplitude = 40

series = seasonality(time, period=period, amplitude=amplitude)

plot_series(time, series)


# define seasonal pattern with upward trend
slope = 0.05
period = 365
amplitude = 40

# generate the data
series = trend(time, slope) + seasonality(time, period, amplitude=amplitude)

plot_series(time, series)


# Noise
def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    noise = rnd.randn(len(time)) * noise_level
    return noise

noise_level = 5
noise_signal = noise(time, noise_level=noise_level, seed=42)

plot_series(time, noise_signal)

series += noise_signal

plot_series(time, series)

