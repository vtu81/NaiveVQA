import sys
import json
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt


def main():
    path = sys.argv[1]
    with open(path, 'r') as fp:
        results=json.load(fp)

    val_acc = np.array(results['accuracy'])

    plt.figure()
    plt.plot(val_acc)
    plt.savefig('val_acc.png')


if __name__ == '__main__':
    main()
