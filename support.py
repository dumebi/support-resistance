import sys
import json
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth


def main():

    dataframe = pandas.read_csv('EURAUD240.csv', names=["Date", "Time", "Open", "High", "Low", "Close", "Volume"])

    Close = dataframe["Close"].values
    Close = Close.reshape(-1, 1)

    # calculate bandwidth (expirement with quantile and samples)
    bandwidth = estimate_bandwidth(Close, quantile=0.1, n_samples=100)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    # fit the data
    ms.fit(Close)

    support = []
    resistance = []
    for k in range(len(np.unique(ms.labels_))):
        my_members = ms.labels_ == k
        values = Close[my_members, 0]    

        # find the edges
        support.append(min(values))
        resistance.append(max(values))

    plt.plot(Close, 'b')
    for i in support:
      plt.axhline(y=i, color='g', linestyle='-')
    for j in resistance:
      plt.axhline(y=j, color='r', linestyle='-')
    plt.title('EURAUD240')
    plt.ylabel('Price')
    plt.show()

if __name__ == "__main__":
    if (len(sys.argv) < 1):
        print('ml.py <inputfile.csv>')
        sys.exit(2)
    main()