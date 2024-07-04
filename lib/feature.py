import networkx as nx
import numpy as np
from scipy import signal as signal_lib

from .helper import agac_cizdir, get_wavelet_coefs


class Feature:
    def display(self, what_to_display="value"):
        agac_cizdir(self.agac, content=what_to_display, title=what_to_display)

    def _preprocess(self, data):
        # pad = int(len(data) / 2)
        # data = list([0] * pad) + list(data) + list([0] * pad)
        # kernel = np.ones(self.mean_kernel_size) / self.mean_kernel_size
        # data = np.convolve(data, kernel, mode="same")
        # hamming_window = signal_lib.blackman(len(data))
        # data = list(np.multiply(data, hamming_window))
        data = np.array(data)
        n = max(abs(data))
        if n > 1:
            data = list((data) / n)
        else:
            data = np.zeros(len(data))
        data = get_wavelet_coefs(data)
        return data

    def compute_dissipated_error(self, output_node, data):
        yol = nx.shortest_path(self.agac, 0, output_node)
        coefs = []
        for n in yol:
            coefs.append(self.agac.nodes[n]["value"])
        err = 0

        for i in range(1, self.branch_depth):
            if len(yol) > i:
                # a= yol[i]

                err += abs(self.agac.nodes[yol[i]]["value"] - data[i])
            # else:
            #     err += abs(0 - data[i])
        # if not  self.learn_mode:
        # print(err, yol, coefs, data[:self.branch_depth])

        return err

    def step(self, data):
        data = self._preprocess(data)
        x = sum(abs(np.array(data[:10])))
        # print(x)
        if x < 0.5:
            return -1, 0
        if self.learn_mode:
            poz = self.add_branch(data[1: self.branch_depth])

        else:
            poz = self.check_branch(data[1: self.branch_depth])
        if poz>0:
            err = self.compute_dissipated_error(poz, data)
        else:
            err=0
        return poz, err

    def add_branch(self, input_data):
        self.modified = True
        poz = 0
        for j, d in enumerate(input_data):  # for all data
            nei = list(self.agac.neighbors(poz))  # get children of node with id poz

            if (
                    len(nei) == 0
                    and self.agac.nodes[poz]["occurrence_count"] > self.node_age_treshold
            ):
                # if there is no node, directly add node
                self.agac.add_node(
                    self.node_counter,
                    range=1,
                    value=d,
                    occurrence_count=1,
                    id=self.node_counter,
                )
                self.agac.add_edge(poz, self.node_counter)
                poz = self.node_counter
                self.node_counter += 1
                break
            else:
                k = -1
                min_dist = 10000000
                for n in nei:
                    distance = abs(self.agac.nodes[n]["value"] - d)
                    if distance < self.agac.nodes[n]["range"] and distance < min_dist:
                        k = n
                        min_dist = distance

                if k >= 0:
                    self.agac.nodes[k]["occurrence_count"] += 1
                    if self.agac.nodes[k]["range"] > self.range_threshold:
                        distance = self.agac.nodes[k]["value"] - d
                        # if poz ==0:
                        #     print(d, "adjust node ",k , self.agac.nodes[k]["value"] ,"value by ", distance,  distance * self.value_delta)
                        self.agac.nodes[k]["value"] -= distance * self.value_delta
                        self.agac.nodes[k]["range"] -= (
                                self.agac.nodes[k]["range"] * self.range_delta
                        )
                    poz = k
                else:
                    if (
                            self.agac.nodes[poz]["occurrence_count"]
                            > self.node_age_treshold
                    ):
                        # if poz == 0:
                        #     print("data = ", d, 'node counter', self.node_counter)
                        #     for n in list(self.agac.neighbors(0)):
                        #         print(self.agac.nodes[n])

                        self.agac.add_node(
                            self.node_counter,
                            range=1,
                            value=d,
                            occurrence_count=1,
                            id=self.node_counter,
                        )
                        self.agac.add_edge(poz, self.node_counter)
                        poz = self.node_counter
                        self.node_counter += 1
                        break

        return poz

    def check_branch(self, input_data):
        depth_factor = 1
        active_nodes = [0]
        active_nodes_error = [0]

        final_nodes = []
        final_nodes_error = []

        for i, d in enumerate(input_data):  # for all data
            temp_active_nodes = []
            temp_active_nodes_error = []

            for j, poz in enumerate(active_nodes):
                nei = list(
                    self.agac.neighbors(poz)
                )  # get neighbours of node with id poz

                if len(nei) == 0:  # if there is no node, directly add node
                    final_nodes.append(poz)
                    final_nodes_error.append(active_nodes_error[j])
                else:
                    dist_list = []
                    for n in nei:
                        dist = abs(self.agac.nodes[n]["value"] - d)
                        dist_list.append(dist)

                    k = min(dist_list)
                    min_item = dist_list.index(k)
                    dist_list.remove(k)

                    if len(dist_list) == 0:
                        continue
                    temp_active_nodes.append(nei[min_item])
                    temp_active_nodes_error.append(
                        active_nodes_error[j] + k * depth_factor
                    )

                    k = min(dist_list)
                    min_item = dist_list.index(k)

                    if (
                            self.agac.nodes[nei[min_item]]["range"]
                            > k - self.range_threshold / 2
                    ):
                        temp_active_nodes.append(nei[min_item])
                        temp_active_nodes_error.append(
                            active_nodes_error[j] + k * depth_factor
                        )

            while len(temp_active_nodes) > 2:
                k = max(temp_active_nodes_error)
                k = temp_active_nodes_error.index(k)
                temp_active_nodes_error.pop(k)
                temp_active_nodes.pop(k)

            if len(temp_active_nodes) == 0:
                break

            active_nodes = temp_active_nodes
            active_nodes_error = temp_active_nodes_error

        return active_nodes.pop(active_nodes_error.index(min(active_nodes_error)))

    def __init__(self, parameters):
        self.agac = nx.DiGraph()
        self.agac.add_node(0, value=999999, range=1, occurrence_count=100, id=-1)
        self.range_delta = 0.01
        self.value_delta = 0.01
        self.node_counter = 1
        self.node_age_treshold = 20
        self.range_threshold = 0.2
        self.path = parameters["path"]
        self.window_size = parameters["window_size"]
        self.learn_mode = parameters["learn_mode"]
        self.branch_depth = parameters["branch_depth"]
        self.modified = False

        self.apply_normalization = False
        self.use_wavelet = True

        self.use_mean_filter = True
        self.mean_kernel_size = 11

        self.apply_hamming = True
        self.hamming_window = signal_lib.hamming(self.window_size)
