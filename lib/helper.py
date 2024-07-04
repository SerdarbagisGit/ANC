import pickle
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pywt import wavedec, WaveletPacket, waverec


def merge_list(input_data, verbose=False):
    if verbose:
        print("merge data")
    return [item for sublist in input_data for item in sublist]


def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(
                iter(nx.topological_sort(G))
            )  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(
            G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None
    ):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def agac_cizdir(agac, title="Tree structure", content="value"):
    plt.rcParams["figure.figsize"] = [15, 10]
    labels = dict((n, round(d[content], 2)) for n, d in agac.nodes(data=True))

    pos = hierarchy_pos(agac, 0)

    plt.title(title + " node values")
    nx.draw_networkx(agac, pos=pos, arrows=True, with_labels=True, labels=labels)
    plt.show()


def agac_cizdir_simple(agac, title="Tree structure", content="value"):
    plt.rcParams["figure.figsize"] = [15, 10]
    labels = dict((n, d[content]) for n, d in agac.nodes(data=True))

    pos = hierarchy_pos(agac, 0)

    plt.title(title + " node values")
    nx.draw_networkx(agac, pos=pos, arrows=True, with_labels=True, labels=labels)
    plt.show()


def get_wavelet_coefs(input_data):
    girdi = np.array(input_data)
    level = int(np.log2(len(girdi)))
    coeff = wavedec(girdi, "haar", level=level)
    return np.round(np.concatenate(coeff), decimals=2)


def get_inverse_wavelet2(coeffs):
    level = int(np.log2(len(coeffs) - 1))
    coeff = [np.array([coeffs[i]]) for i in range(level + 2)]
    for i in range(2, level + 2):
        coeff[i] = coeffs[2 ** (i - 1):2 ** i]
    data = waverec(coeff, "haar")
    return data


def get_wavelet_coefs1(wave):
    # wave = wave - np.mean(wave)
    wp = WaveletPacket(data=wave, wavelet="haar", mode="symmetric")
    return np.squeeze(
        [wps.data for wps in wp.get_level(int(np.log2(len(wave))), "natural")]
    )


def add_noise(data, mu=0.0, sigma=1.0, verbose=False):
    noise = np.random.normal(mu, sigma, [len(data)])
    a = np.array(data)
    a = np.sum(a ** 2)
    b = np.sum(noise ** 2)
    if verbose:
        if b != 0:
            print("xxSNR =", 10 * np.log10((a - b) / b))
        else:
            print("we have 0 data ")
    return list(data + noise)


def save_signal(data, path):
    with open(path, "wb") as f:
        np.save(f, np.array(data))


def restore_signal(path):
    with open(path, "rb") as f:
        data = np.load(f)
    return list(data)


def save_object(obj, obj_path):
    pickle.dump(obj, open(obj_path, "wb"))


def restore_object(obj_path):
    obj = pickle.load(open(obj_path, "rb"))
    return obj


def pad_audio_end(samples, L):
    if len(samples) >= L:
        return samples
    else:
        return np.pad(
            samples,
            pad_width=(0, L - len(samples)),
            mode="constant",
            constant_values=(0, 0),
        )


def pad_audio_beginning(samples, L):
    if len(samples) >= L:
        return samples
    else:
        return np.pad(
            samples,
            pad_width=(L - len(samples), 0),
            mode="constant",
            constant_values=(0, 0),
        )


def shufle_list(data, seed=101):
    x = data.copy()
    random.Random(seed).shuffle(x)
    return x


def shuffle_list_with_label(data, label, seed=101):
    temp = list(zip(data, label))
    random.Random(seed).shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    res1, res2 = list(res1), list(res2)
    return res1, res2


def extend_ops(ops):
    random.seed(42)
    extra_ops = []
    extra = random.sample(range(1, len(ops)), int(len(ops) / 2))
    for i in range(len(extra)):
        extra_ops.append(ops[i])

    random.seed(43)
    extra = random.sample(range(1, len(ops)), int(len(ops) / 2))
    for i in range(len(extra)):
        extra_ops.append(ops[i])
    print(extra_ops[0], extra_ops[-1])

    t_ops = ops + extra_ops
    return t_ops


def to_signal(opss,  env,  L, labels=False, pad=0):
    ops = opss.copy()
    all_label = []
    all_wave = []
    random.seed(42)
    x = [random.randint(0, 8) for p in range(len(ops))]

    for i, ex in enumerate(ops):
        # clean  operation
        if labels:
            expr = ex[:-1]
        else:
            expr = ex
        signal = env.gen_signal_for_symbols(expr, add_blanks=True)
        if pad == 0:
            signal = pad_audio_end(signal, L)
        else:
            signal = pad_audio_beginning(signal, L)

        all_label.append(str(ex[-1]))
        all_wave.append(signal)
    return all_wave, all_label


def to_signal1(opss,env,  L, labels=False, pad=0):
    ops = opss.copy()
    all_label = []
    all_wave = []
    for i, ex in enumerate(ops):
        # clean  operation
        if labels:
            expr = ex[:-1]
        else:
            expr = ex
        signal = env.gen_signal_for_symbols(expr, add_blanks=True)
        if pad == 0:
            signal = pad_audio_end(signal, L)
        else:
            signal = pad_audio_beginning(signal, L)

        all_label.append(str(ex[-1]))
        all_wave.append(signal)
    return all_wave, all_label


def randomly_insert_blanks(data, spaces):
    d = data.copy()
    for i in spaces:
        # r= random.randint(0,4+i)
        d.insert(i, 0)
    return d


def convert_expression_to_nice(exp):
    rez = []
    for e in exp:
        if e == 0:
            rez.append("s")
        elif e < 11:
            rez.append(str(e - 1))
        elif e == 11:
            rez.append("+")
        elif e == 12:
            rez.append("=")
        elif e == 13:
            rez.append("-")
    return rez