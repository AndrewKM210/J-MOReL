import pandas as pd
from tabulate import tabulate


class Log:
    def __init__(self):
        self._log = {}
        self._step = 0

    def log(self, key, value):
        if key not in self._log.keys():
            self._log[key] = [value]
        else:
            self._log[key].append(value)

    def step(self):
        self._step += 1
        for k, v in self._log.items():
            if len(self._log[k]) < self._step:
                self._log[k] += [self._log[k][-1]] * (self._step - len(self._log[k]))

    def get_current_log(self):
        current_log = {}
        for k, v in self._log.items():
            current_log[k] = self._log[k][-1]
        return current_log

    def print_current_log(self, sort=True):
        if sort:
            print(tabulate(sorted(self.get_current_log().items())))
        else:
            print(tabulate(self.get_current_log().items()))

    def save(self, path):
        pd.DataFrame(self._log).to_csv(path, index_label="step")
