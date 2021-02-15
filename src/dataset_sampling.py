# %%
import os
import pickle
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import minmax_scale

from itertools import combinations, permutations, product
from scipy.stats import norm
from numpy import random

# %%
DATA_PATH = '../data/'
# %%
df = pd.read_json(os.path.join(DATA_PATH, '/dev/cluster_12_cats.json'))
df.info()

# %%
df['category'].unique()

# %%


def sample_sigmoid(time_range, n=1, change_rate=0.5):
    x = np.arange(time_range)
    mid = int(time_range / 2)
    y = (1 / (1 + np.exp(-0.1 * (x - mid)))) * n * change_rate
    y = (y - y.min()) / (y.max() - y.min())

    y = n + y * n * change_rate

    # print(y)
    plt.figure(figsize=(20, 5))
    plt.plot(x, y)
    plt.show()


def sample_bell(time_range, n=1, change_rate=0.5, std=0):
    x = np.arange(time_range)
    mu = int(time_range / 2)
    std = std if std else int(time_range / 5)
    y = norm.pdf(x, mu, std)
    # scale 0-1
    y_min = y.min()
    y_max = y.max()
    y = (y - y_min) / (y_max - y_min)
    # add n docs
    y = n + y * n * change_rate

    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.show()


def sample_linear(time_range, n=100, change_rate=0.5):
    x = np.arange(time_range)
    y = (x - x.min()) / (x.max() - x.min())
    y = n + y * n * change_rate

    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.show()


# %%
sample_sigmoid(50, n=1, change_rate=0.5)

# %%


def linear_pattern(n=1, start=0, stop=100, change_rate=1):
    """
    Sampling up pattern, start and end in random points
    """
    # print(lower_p, upper_p)
    # change_points = np.array([0, lower_p, upper_p, timeline], dtype=int)
    x = np.arange(start, stop)
    # normalize x to range 0-1
    y = (x - start) / (stop - start)
    freq_rates = n + y * n * change_rate

    return freq_rates


def sigmoid_pattern(n=1, start=0, stop=100, change_rate=1):
    x = np.arange(start, stop)
    mid = int((stop - start) / 2)
    y = 1 / (1 + np.exp(-0.1 * (x - mid)))
    y = (y - y.min()) / (y.max() - y.min())

    freq_rates = n + y * n * change_rate
    return freq_rates


def flat_pattern(n=1, start=0, stop=100):
    freq_rates = np.ones(stop - start) * n
    return freq_rates


def bell_pattern(n=1, start=0, stop=100, change_rate=1, std=0):
    time_range = stop - start

    x = np.arange(time_range)
    mu = int(time_range / 2)

    std = std if std else int(time_range / 5)
    y = norm.pdf(x, mu, std)
    # scale 0-1
    y = (y - y.min()) / (y.max() - y.min())
    # add n docs
    freq_rates = n + y * n * change_rate

    return freq_rates


def sample_pattern(pattern, data, n_doc, timeline=100, change_rate=0.01):
    sample = None

    if pattern == 'up':
        lower_p = np.random.randint(low=1, high=timeline - 30)
        upper_p = np.random.randint(low=lower_p + 20, high=timeline)

        # f1, f2, f3 [-1] is the start of freqs ratio for the pattern as the chaning variable
        f1 = flat_pattern(1, start=0, stop=lower_p)
        f2 = sigmoid_pattern(f1[-1], start=lower_p,
                             stop=upper_p, change_rate=change_rate)
        f3 = flat_pattern(f2[-1], start=upper_p, stop=timeline)

        # the frequency ratio
        time_freqs = np.concatenate((f1, f2, f3))
        time_freqs = time_freqs / time_freqs.sum()

        change_points = np.array([lower_p, upper_p])

    elif pattern == 'down':
        lower_p = np.random.randint(low=1, high=timeline - 30)
        upper_p = np.random.randint(low=lower_p + 20, high=timeline)

        f1 = flat_pattern(1, start=0, stop=lower_p)
        f2 = sigmoid_pattern(f1[-1], start=lower_p,
                             stop=upper_p, change_rate=-change_rate)
        f3 = flat_pattern(f2[-1], start=upper_p, stop=timeline)

        # the frequency ratio
        time_freqs = np.concatenate((f1, f2, f3))
        time_freqs = time_freqs / time_freqs.sum()

        change_points = np.array([lower_p, upper_p])

    elif pattern == 'spike_up':
        n_point = np.random.randint(1, 5)
        invalid = True

        while invalid:
            change_points = np.sort(np.random.choice(
                range(5, timeline - 5), n_point, replace=False))
            diff = np.diff(change_points)
            invalid = len(np.where(diff < 10)[0])

        change_rates = np.random.uniform(0.3, change_rate, n_point)

        cur_p = 0
        cur_n = 1

        time_freqs = []

        for i, p in enumerate(change_points):
            # print(cur_p, p - 2)
            f1 = flat_pattern(cur_n, start=cur_p, stop=p - 2)
            cur_n = f1[-1]
            f2 = bell_pattern(cur_n, start=p - 2, stop=p + 3,
                              change_rate=change_rates[i], std=0.1)
            cur_n = f2[-1]

            time_freqs.append(f1)
            time_freqs.append(f2)

            cur_p = p + 3

            if i == len(change_points) - 1:
                f3 = flat_pattern(cur_n, start=cur_p, stop=timeline)
                time_freqs.append(f3)

        time_freqs = np.concatenate(time_freqs)
        time_freqs = time_freqs / time_freqs.sum()

    elif pattern == 'spike_down':
        n_point = np.random.randint(1, 5)
        invalid = True
        # generate n points with min distance 10
        while invalid:
            change_points = np.sort(np.random.choice(
                range(5, timeline - 5), n_point, replace=False))
            diff = np.diff(change_points)
            invalid = len(np.where(diff < 10)[0])

        change_rates = np.random.uniform(0.3, change_rate, n_point)
        cur_p = 0
        cur_n = 1

        time_freqs = []

        for i, p in enumerate(change_points):

            f1 = flat_pattern(cur_n, start=cur_p, stop=p - 2)
            cur_n = f1[-1]
            f2 = bell_pattern(cur_n, start=p - 2, stop=p + 3, change_rate=-change_rates[i], std=0.1)
            cur_n = f2[-1]

            time_freqs.append(f1)
            time_freqs.append(f2)

            cur_p = p + 3

            if i == len(change_points) - 1:
                f3 = flat_pattern(cur_n, start=cur_p, stop=timeline)
                time_freqs.append(f3)

        time_freqs = np.concatenate(time_freqs)
        time_freqs = time_freqs / time_freqs.sum()

    elif pattern == 'up_down':
        lower_p = np.random.randint(low=1, high=timeline - 20)
        upper_p = np.random.randint(low=lower_p + 10, high=timeline)

        f1 = flat_pattern(1, start=0, stop=lower_p)
        f2 = bell_pattern(f1[-1], start=lower_p, stop=upper_p, change_rate=change_rate)
        f3 = flat_pattern(f2[-1], start=upper_p, stop=timeline)

        # the frequency ratio
        time_freqs = np.concatenate((f1, f2, f3))
        time_freqs = time_freqs / time_freqs.sum()

        mid_p = int(lower_p + (upper_p - lower_p) / 2)
        change_points = np.array([lower_p, mid_p, upper_p])

    elif pattern == 'down_up':
        lower_p = np.random.randint(low=1, high=timeline - 20)
        upper_p = np.random.randint(low=lower_p + 10, high=timeline)

        f1 = flat_pattern(1, start=0, stop=lower_p)
        f2 = bell_pattern(f1[-1], start=lower_p, stop=upper_p, change_rate=-change_rate)
        f3 = flat_pattern(f2[-1], start=upper_p, stop=timeline)

        # the frequency ratio
        time_freqs = np.concatenate((f1, f2, f3))
        time_freqs = time_freqs / time_freqs.sum()

        mid_p = int(lower_p + (upper_p - lower_p) / 2)
        change_points = np.array([lower_p, mid_p, upper_p])

    else:
        time_freqs = flat_pattern(1, start=0, stop=timeline)
        time_freqs = time_freqs / time_freqs.sum()
        change_points = np.empty(shape=(0,))

    # calculate the docs_num based on the total docs and its freqs distribution
    docs_num = (n_doc * time_freqs).astype(int)
    sample = data.sample(n_doc)
    sample['time'] = -1

    cur = 0
    for i, n in enumerate(docs_num):
        sample.iloc[cur: cur + n, sample.columns.get_loc("time")] = i
        cur += n

    # because the freq is converted to int so some articles will remain -1 for time, we need to prunt those
    sample = sample[sample['time'] > -1]

    return sample, change_points.astype(int)


def create_samples(df, timeline=100, n_samples=100, min_doc=50, max_doc=100, frac=0.99, change_rate=0.5):
    categories = df['category'].unique()

    samples = []   # list article ids
    tracker = pd.DataFrame(columns=['category', 'event', 'pivots'])
    # sample_pivots = []  # list of pivots index in timeline, need to map with ids
    patterns = ['up', 'down', 'up_down', 'down_up',
                'spike_up', 'spike_down', 'noise']
    # patterns = ['spike_down']

    g = df.groupby(['category'])

    for _ in range(n_samples):
        # select random category as the target
        # And the rest as noise
        cat = np.random.choice(categories)
        event = np.random.choice(patterns)

        df_cat = g.get_group(cat)
        df_len = len(df_cat)
        n_doc = np.random.randint(min_doc, max_doc)
        n_doc = min(n_doc, df_len)

        df_sample, points = sample_pattern(
            event, df_cat[['id', 'category']], n_doc, change_rate=change_rate)
        tracker = tracker.append(
            {'category': cat, 'event': event, 'pivots': points}, ignore_index=True)

        df_sample = df_sample.sample(frac=frac)
        df_sample.reset_index(drop=True, inplace=True)
        samples.append(df_sample)

    return samples, tracker


# %%
samples, tracker = create_samples(df, timeline=100, n_samples=2000, min_doc=5000, max_doc=15000, change_rate=0.8)
