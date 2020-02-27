import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import glob
import re
import numpy as np
import os

import pandas as pd

def scrape_file(file: str, pattern: str, one_per_line: bool = True,
                data_type=float):
    data = []
    with open(file, "r") as ff:
        lines = ff.readlines()
        for line in lines:
            matches = re.findall(pattern, line)
            if len(matches) <= 0:
                continue

            if one_per_line:
                matches = [matches[0]]
            data += [data_type(m) for m in matches]
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot pattern from regex.')
    parser.add_argument('files', metavar='N', type=str, nargs='+',
                        help='list of files to scrape')

    args = parser.parse_args()
    data_files = []

    for f, name in zip(args.files[1::2], args.files[0::2]):
        data_files.append((f, name))
        assert os.path.isfile(f), f"{f} is not a file."

    pattern = '\(\'rel_pos\', ([^)]*)'

    rolling_len = 15

    fig, ax = plt.subplots()

    for i, (f, name) in enumerate(data_files):
        d = scrape_file(f, pattern)

        steps = np.arange(len(d))
        value = pd.Series(np.array(d) / 10)

        win = value.rolling(rolling_len)
        mu = win.mean()
        sigma = win.std()

        base_line, = ax.plot(steps, mu, label=f"{name}")
        # ax.fill_between(steps, mu + sigma, mu - sigma,
        #                 facecolor=base_line.get_color(), alpha=0.5)

    ax.legend(loc="upper right")
    plt.title('1 step relative pose (rolling window 15)')
    plt.show()





