import random

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as signal_lib

from .helper import merge_list, pad_audio_beginning


class Veri:

    def gen_symbols_more_complex(
        self, symbol_lengths, max_freq, max_amp, verbose=False
    ):
        sample = {}
        sample["blank"] = {"p": 0.2, "val": np.array([0] * symbol_lengths[0])}

        signals = []

        index = 1
        if verbose:
            print("name\tf_m\ta_m\tf_c\t_a_c")
        for i in range(2, 7, 3):
            for j in range(2, 7, 2):
                for k in range(1, 7, 2):
                    symbol_length = symbol_lengths[index]
                    t = np.linspace(0, 1, symbol_length)
                    index += 1
                    A_m, f_m = i, i + k
                    A_c, f_c = 10, 16 - j - k

                    message = A_m * np.cos(2 * np.pi * f_m * t)
                    a = (A_c + message) * np.sin(2 * np.pi * f_c * t)
                    if verbose:
                        print("symb_" + str(index - 1), symbol_length, f_m, A_m, f_c, A_c)

                    sample["symb_" + str(index - 1)] = {"p": 0.2, "val": a}
                    if index == len(symbol_lengths):
                        return sample
        return sample


    def genSignalWithGivenSymbols(
        self, symbol_list, add_blanks=False, add_noise=False, verbose=False
    ):
        names = list(self.symbols.keys())

        signal = []
        for s in symbol_list:
            if verbose:
                print(names[s], self.symbols[names[s]]["val"])

            if s == 0:
                signal = [*signal, *self.symbols["blank"]["val"]]
            else:
                signal = [*signal, *self.symbols[names[s]]["val"]]
            if add_blanks:
                signal = [*signal, *self.symbols["blank"]["val"]]
        if add_noise:
            signal = add_noise(signal, self.mu, self.sigma)
        return signal

    def genPaddedSignalWithGivenSymbols(
        self, symbol_list, add_blanks=False, add_noise=False, verbose=False
    ):
        names = list(self.symbols.keys())
        resulting_symbol = -1
        for i in range(len(symbol_list)):
            if symbol_list[i] != 0:
                resulting_symbol = i

        signal = []
        for i in range(resulting_symbol):
            s = symbol_list[i]
            if verbose:
                print(names[s], self.symbols[names[s]]["val"])

            if s == 0:
                signal = [*signal, *self.symbols["blank"]["val"]]
            else:
                signal = [*signal, *self.symbols[names[s]]["val"]]
            if add_blanks:
                signal = [*signal, *self.symbols["blank"]["val"]]
        signal = pad_audio_beginning(signal, 3200)
        for i in range(resulting_symbol, len(symbol_list)):
            s = symbol_list[i]
            if verbose:
                print(names[s], self.symbols[names[s]]["val"])

            if s == 0:
                signal = [*signal, *self.symbols["blank"]["val"]]
            else:
                signal = [*signal, *self.symbols[names[s]]["val"]]
            if add_blanks:
                signal = [*signal, *self.symbols["blank"]["val"]]

        if add_noise:
            signal = add_noise(signal, self.mu, self.sigma)

        return signal

    def __init__(self, symbol_lengths=[8, 8, 8, 8]):
        self.symbol_lengths = symbol_lengths
        self.symbols = ""
        self.blank_len = 4  # symbol_lengths[0]
        self.mu = 0
        self.sigma = 0.5


class Environment:
    def gen_signal_for_symbols(
        self, symbols, add_blanks=False, add_noise=False, verbose=False
    ):
        signal = self.veri.genSignalWithGivenSymbols(
            symbols, add_blanks, add_noise=add_noise, verbose=verbose
        )
        return signal

    def display_symbols(self, which_symbols=[], verbose=False):
        plt.rcParams["figure.figsize"] = [15, 10]
        names = list(self.veri.symbols.keys())
        for n, i in enumerate(names):
            if verbose:
                print(i, self.veri.symbols[i]["val"][:])
            if n in which_symbols or len(which_symbols) == 0:
                plt.plot(self.veri.symbols[i]["val"], label=i)
            if n % 5 == 4:
                plt.legend()
                plt.show()

        if len(names) % 5 != 0:
            plt.legend()
            plt.show()

    def _gen_main_ops(self):
        l = []
        for i in range(10):
            for j in range(10):
                if i + j < 10:
                    # l.append([i+1, 11, j+1, 12, i + j+1])
                    l.append([i + 1, 11, j + 1, 12, i + j + 1])
        for i in range(10):
            for j in range(10):
                if i - j >= 0:
                    # l.append([i+1, 11, j+1, 12, i + j+1])
                    l.append([i + 1, 13, j + 1, 12, i - j + 1])
        return l

    def gen_ops_signal(
        self, expression_count, add_blanks=False, add_noise=False, verbose=False
    ):
        l = []
        rez = []
        for i in range(expression_count):
            # l.append([-1] * 5)
            l.append(self.all_ops[random.randint(0, len(self.all_ops) - 1)])
            l.append([0] * 5)
        l = merge_list(l)
        rez.append(
            self.veri.genSignalWithGivenSymbols(
                symbol_list=l, add_blanks=add_blanks, add_noise=add_noise, verbose=verbose
            )
        )

        rez = merge_list(rez)
        return l, rez

    def gen_split_ops_train_test(self, test_percentage=0.2):
        train_ops = self.all_ops.copy()
        test_ops = []
        test_size = int(len(self.all_ops) * test_percentage)
        all_size = len(self.all_ops)
        random.seed(42)
        for i in range(test_size):
            temp = random.sample(range(0, all_size), 1)[0]
            all_size -= 1
            test_ops.append(train_ops.pop(temp))
        return train_ops, test_ops

    def __init__(self, signal_conf):
        print("Environment initiated")
        self.all_ops = self._gen_main_ops()
        self.train_ops, self.test_ops = self.gen_split_ops_train_test(
            test_percentage=0.2
        )

        length_of_symbols = [512] * signal_conf["number_of_symbols"]

        if signal_conf["variable_len_symbols"]:
            length_of_symbols = [
                256,
                576,
                448,
                640,
                512,
                384,
                512,
                448,
                576,
                640,
                512,
                448,
                384,
                512,
                640,
                512,
            ]
            # for l in range(len(length_of_symbols)):
            #     length_of_symbols[l] = random.randrange(512 - 32 * 4, 512 + 32 * 4, 32)
            # length_of_symbols[0] = 256 + 128
        # length_of_symbols = [384, 576, 480, 448, 480, 384, 544, 480, 480, 544, 544, 576, 544, 608, 576]

        print(length_of_symbols)
        self.veri = Veri(symbol_lengths=length_of_symbols)
        self.veri.symbols = self.veri.gen_symbols_more_complex(
            symbol_lengths=length_of_symbols, max_freq=16, max_amp=10, verbose=False
        )
