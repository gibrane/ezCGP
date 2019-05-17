import numpy as np
import matplotlib.pyplot as plt

class argInt():

    def __init__(self, value=None, stdev_perc=0.1):
        if value is not None:
            self.value = int(value)
        else:
            self.value = np.random.randint(low=0, high=int(1e3+1))
        self.stdev_perc = stdev_perc
        self.mutate()

    def __repr__(self):
        return str(self.value)

    def mutate(self):
        loc = self.value
        scale = max(loc*self.stdev_perc, 5)
        self.value = round(np.random.normal(loc, scale))


class argBoundedFloat():

    def __init__(self, low=-1e3, high=1e3, value=None):
        self.low = int(low)
        self.high = int(high)
        if value is not None:
            self.value = value
        else:
            self.mutate()

    def __repr__(self):
        return str(self.value)

    def mutate(self):
        self.value = np.random.uniform(self.low, self.high)



class argBoundedInt():

    def __init__(self, low=-1e3, high=1e3, value=None):
        self.low = int(low)
        self.high = int(high)
        if value is not None:
            self.value = round(value)
        else:
            self.mutate()

    def __repr__(self):
        return str(self.value)

    def mutate(self):
        self.value = round(np.random.uniform(self.low, self.high))



class argPow2():

    def __init__(self, low=1, high=8, pow2=None):
        self.low = int(low)
        self.high = int(high)
        if pow2 is not None:
            self.value = 2**int(pow2)
        else:
            self.mutate()

    def __repr__(self):
        return str(self.value)

    def mutate(self):
        pow2 = np.random.randint(self.low, self.high+1)
        self.value = int(2**pow2)

#generations = 100
#fig, axes = plt.subplots(nrows=2, ncols=generations)
