import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner
from keras_tuner import RandomSearch


def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.figure(figsize=(10, 6))
    if type(series) is tuple:
        for series_num in series:
            plt.plot(time[start:end], series_num[start:end], format)
    else:
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


# Noise
def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    noise = rnd.randn(len(time)) * noise_level
    return noise


time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
amplitude = 20
slope = 0.09
noise_level = 5
period = 365

series = baseline + trend(time, slope)
series += seasonality(time, period=period, amplitude=amplitude)
series += noise(time, noise_level=noise_level, seed=42)


plot_series(time, series)


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(
        lambda window: (window[:-1], window[-1])
    )
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)


def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(
        units=hp.Int('units', min_value=10, max_value=30, step=2),
        activation="relu", input_shape=[window_size]
    ))
    model.add(tf.keras.layers.Dense(10, activation="relu"))
    model.add(tf.keras.layers.Dense(1))

    model.compile(loss="mse",
                  optimizer=tf.keras.optimizers.SGD(
                      hp.Choice('momentum',
                                values=[.9, .7, .5, .3]),
                      learning_rate=1e-5
                  ))
    return model

tuner = RandomSearch(build_model,
                     objective='loss', max_trials=150,
                     executions_per_trial=3, directory='my_dir',
                     project_name='hello')

tuner.search(dataset, epochs=100, verbose=0)

print(tuner.results_summary())
#
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(units=hp., activation="relu", input_shape=[window_size]),
#     tf.keras.layers.Dense(10, activation="relu"),
#     tf.keras.layers.Dense(1)
# ])
#
#
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: 1e-8 * 10 ** (epoch / 20)
# )
#
# optimizer = tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9)
# model.compile(loss="mse", optimizer=optimizer)
#
# history = model.fit(dataset, epochs=100,
#           callbacks=[lr_schedule], verbose=0)
#
# lrs = 1e-8 * (10 ** (np.arange(100) / 20))
# plt.semilogx(lrs, history.history["loss"])
# plt.axis([1e-7, 1e-4, 0, 300])
# plt.show()


# forecast = []
# for time in range(len(series) - window_size):
#     forecast.append(
#         model.predict(series[time:time + window_size][np.newaxis])
#     )
#
# forecast = forecast[split_time - window_size:]
# results = np.array(forecast).squeeze()
#
# plot_series(time_valid, (x_valid, results))
# print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())


