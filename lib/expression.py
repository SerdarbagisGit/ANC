import networkx as nx

from .helper import agac_cizdir


class Expression:
    def display(self, what_to_display="value"):
        agac_cizdir(self.agac, content=what_to_display, title=what_to_display)

    def step(self, data, verbose=False):
        if self.learn_mode:
            return self.train(data, verbose=verbose)
        elif self.counter > 1:
            return self.test(data, verbose=verbose)
        else:
            return None

    def train(self, input_symbol, verbose=False):
        # evolve
        if input_symbol >= 0:
            if not self.learn_mode:
                end_state = False
                for poz in self.aktive_nodes:
                    if input_symbol == self.agac.nodes[poz]["value"]: # inhibit
                        # print("we have an existing item in aktive list")
                        # self.agac.nodes[poz]["age"] = 0
                        self.aktive_nodes.remove(poz)
                        return
                    if not list(self.agac.neighbors(poz)):
                        end_state = True
                if end_state: # trigger output
                    # self.aktive_nodes=[self.aktive_nodes[-1]]
                    # print("we have end state", self.aktive_nodes[-1], input_symbol)
                    self.aktive_nodes = [0] + self.agac.nodes[self.aktive_nodes[-1]]["equivalences"]
                    for n in self.aktive_nodes[1:]:
                        self.agac.nodes[n]["age"] = self.forget_period

                    # print(self.aktive_nodes)

            next_aktive_nodes = [0]
            for poz in self.aktive_nodes:
                if poz == 0 or self.agac.nodes[poz]["age"] > 0:
                    nei = list(self.agac.neighbors(poz))
                    k = -1
                    for n in nei:
                        if self.agac.nodes[n]["value"] == input_symbol:
                            k = n
                            break
                    if k >= 0:
                        if self.agac.nodes[k]["occurrence_count"] > 2:
                            next_aktive_nodes.append(k)
                        self.agac.nodes[k]["age"] = self.forget_period
                        self.agac.nodes[k]["occurrence_count"] += 1

                    else:
                        if self.learn_mode:
                            next_ = []
                            if len(next_aktive_nodes) > 1:
                                next_ = [next_aktive_nodes[1]]
                            self.agac.add_node(
                                self.counter,
                                value=input_symbol,
                                occurrence_count=1,
                                age=self.forget_period,
                                id=self.counter,
                                equivalences=next_,
                            )
                            self.agac.add_edge(poz, self.counter)
                            # next_aktive_nodes.append(self.counter)
                            self.counter += 1
            self.aktive_nodes = next_aktive_nodes

        # clean inactive nodes while training
        if self.learn_mode:
            for poz in self.aktive_nodes[1:]:
                self.agac.nodes[poz]["age"] += -1
                if self.agac.nodes[poz]["age"] <= 0 and poz > 0:
                    self.aktive_nodes.remove(poz)

        # learn connections, predict and activate
        if input_symbol >= 0 and len(self.aktive_nodes) > 1:
            # find min aktive node
            min_aktive_nodes_occurences = 100000000
            for i in self.aktive_nodes:
                if min_aktive_nodes_occurences > self.agac.nodes[i]["occurrence_count"]:
                    min_aktive_nodes_occurences = self.agac.nodes[i]["occurrence_count"]

            # learn connections
            if self.learn_mode and min_aktive_nodes_occurences > 5:
                for i in range(len(self.aktive_nodes) - 1, 0, -1):
                    source = self.agac.nodes[self.aktive_nodes[i]]
                    for j in range(1, i):
                        target = self.agac.nodes[self.aktive_nodes[j]]
                        if source["occurrence_count"] < target["occurrence_count"]:
                            # print(i, j, self.aktive_nodes[j], self.aktive_nodes[i])
                            if (
                                    not self.aktive_nodes[j]
                                        in self.agac.nodes[self.aktive_nodes[i]]["equivalences"]
                            ):
                                self.agac.nodes[self.aktive_nodes[i]][
                                    "equivalences"
                                ].append(self.aktive_nodes[j])
            # predict next nodes
            if not self.learn_mode:
                if verbose:
                    print("min_aktive_nodes_occurences", min_aktive_nodes_occurences)

                for poz in self.aktive_nodes:
                    nei = self.agac.neighbors(poz)
                    for n in nei:
                        if (
                                self.agac.nodes[n]["occurrence_count"]
                                / self.agac.nodes[poz]["occurrence_count"]
                                > 0.8
                                and self.agac.nodes[n]["occurrence_count"] > 10
                                and not n in self.aktive_nodes
                        ):
                            if verbose:
                                print(poz, "added", n)
                                print(self.agac.nodes[poz]["value"], "added", self.agac.nodes[n]["value"])
                            self.aktive_nodes.append(n)
                            self.agac.nodes[n]["age"] = self.forget_period
                            if verbose:
                                print("added equivalences")
                            for eq in self.agac.nodes[n]["equivalences"]:
                                if eq not in self.aktive_nodes:
                                    if verbose:
                                        print(" ", n, "equi added", eq)
                                        print(" ", self.agac.nodes[n]["value"], "added", self.agac.nodes[eq]["value"])
                                    self.aktive_nodes.append(eq)
                                    self.agac.nodes[eq]["age"] = self.forget_period
                                else:
                                    self.agac.nodes[eq]["age"] = self.forget_period
                        elif n in self.aktive_nodes:
                            self.agac.nodes[n]["age"] = self.forget_period

    def test(self, input_symbol, verbose=False):
        result = []
        # clean inactive nodes
        for poz in self.aktive_nodes[1:]:
            self.agac.nodes[poz]["age"] += -1
            if self.agac.nodes[poz]["age"] <= 0 and poz > 0:
                self.aktive_nodes.remove(poz)

        if input_symbol >= 0:
            data_flow = False
            if len(self.input_buffer)!=0:
                nei = self.agac.neighbors(self.aktive_nodes[-1])
                for n in nei:
                    if self.agac.nodes[n]["value"] == input_symbol:
                        data_flow = True
                        break
            else:
                data_flow=True


            if self.input_buffer:
                if self.input_buffer[-1] != input_symbol:
                    self.input_buffer.append(input_symbol)
            else:
                self.input_buffer.append(input_symbol)

            if not data_flow:
                # print("switch", self.aktive_nodes, self.input_buffer)
                self.input_buffer.reverse()
                # print("switch", self.aktive_nodes, self.input_buffer)
                self.aktive_nodes=[0]
                for n in self.input_buffer:
                    self.train(n)
                    # print("step",n, self.aktive_nodes)
                self.input_buffer = []
            else:
                self.train(input_symbol)

            if verbose:
                print(
                    "time:",
                    "\tSybol:",
                    input_symbol,
                    "\t active nodes",
                    self.aktive_nodes,
                )
            for n in self.aktive_nodes:
                nei = list(self.agac.neighbors(n))
                # print(n)
                max1 = 0
                max_id = 0
                summ = 0
                for n1 in nei:
                    if self.agac.nodes[n1]["occurrence_count"] > max1:
                        max1 = self.agac.nodes[n1]["occurrence_count"]
                        max_id = n1
                    summ += self.agac.nodes[n1]["occurrence_count"]
                if summ == 0:
                    summ = 1
                if verbose:
                    print(
                        "node ",
                        n,
                        "the most probable next node is",
                        max_id,
                        "probability = ",
                        round(max1 / summ, 2),
                        "\n\tcurrent",
                        self.agac.nodes[n],
                        "\n\tnext",
                        self.agac.nodes[max_id],
                    )
                if max1 / summ > 0.8:
                    if max_id not in self.aktive_nodes:
                        self.aktive_nodes.append(max_id)
                        self.agac.nodes[max_id]["age"] = self.forget_period
                    if self.agac.nodes[max_id]["value"] != input_symbol:
                        result.append(max_id)
                    if verbose:
                        print("next node is ", max_id)

        res = [self.agac.nodes[x]["value"] for x in result]
        res = list(dict.fromkeys(res))
        if res:
            self.input_buffer = []

        if len(self.pred) > 0:
            if result != self.pred:
                result = self.pred
        self.pred = result
        return res  # result

    def most_frequent(self, List):
        return max(set(List), key=List.count)

    def __init__(self, path="", learn_mode=True, forget_period=20):
        self.counter = 1
        self.agac = nx.DiGraph()
        self.agac.add_node(
            0,
            value=999999,
            occurrence_count=100000000,
            id=0,
            age=0,
            equivalences=[],
            active=False,
        )
        self.aktive_nodes = [0]
        # self.aktive_nodes_occur = [0]
        self.input_buffer = []
        self.predicted_nodes = []
        self.path = path
        self.learn_mode = learn_mode
        self.forget_period = forget_period
        self.pred = []
