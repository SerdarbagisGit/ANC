import networkx as nx


class Symbol:
    def display(self, what_to_display="value"):
        labels = dict((n, d["value"]) for n, d in self.agac.nodes(data=True))
        nx.draw(self.agac, arrows=True, with_labels=True, labels=labels)

    def step(self, data, verbose=False):
        if self.learn_mode:
            return self.train(
                data, verbose=verbose
            )  # self.add_branch(data[1: self.branch_depth])
        elif self.node_counter > 1:
            return self.test(
                data, verbose=verbose
            )  # self.check_branch(data[1: self.branch_depth])
        else:
            return None

    def train(self, input_feature, verbose=False):
        new_node = 0
        # reset from input
        if input_feature < 0:
            self.active_nodes = []
        else:
            # get children of the root
            nei = list(self.agac.neighbors(0))
            feature_node = [n for n in nei if self.agac.nodes[n]["value"] == input_feature]

            # if the feature is available
            if len(feature_node) > 0:
                f_node = feature_node[0]
                # if not mature .. grow
                if self.agac.nodes[f_node]["occurrence_count"] < self.growth_threshold:
                    self.agac.nodes[f_node]["occurrence_count"] += 1

                # if mature
                else:
                    self.agac.nodes[f_node]["occurrence_count"] += 1
                    new_node = f_node

            else:
                # print("added node", self.node_counter, input_feature)
                self.agac.add_node(
                    self.node_counter,
                    value=input_feature,
                    occurrence_count=1,
                    id=self.node_counter,

                    energy=0,
                    depth=1,
                )
                self.agac.add_edge(0, self.node_counter)
                self.node_counter += 1

            temp_active_nodes = []
            if new_node > 0:
                temp_active_nodes.append(new_node)

                for l, prev_node in enumerate(self.active_nodes):
                    if new_node == prev_node:
                        temp_active_nodes = self.active_nodes
                        break

                    nei_c = list(self.agac.neighbors(new_node))
                    nei_p = list(self.agac.neighbors(prev_node))
                    common_node = list(set(nei_c) & set(nei_p))

                    if len(common_node) > 0:
                        self.agac.nodes[common_node[0]]["occurrence_count"] += 1

                        if (
                                self.agac.nodes[common_node[0]]["occurrence_count"]
                                > self.growth_threshold
                        ):
                            temp_active_nodes.append(common_node[0])
                            new_node = common_node[0]
                        else:
                            break
                    elif prev_node > 0:
                        self.agac.add_node(
                            self.node_counter,
                            value=new_node * 1000000 + prev_node,
                            occurrence_count=1,
                            id=self.node_counter,

                            energy=self.active_threshold,
                            depth=l + 2,
                        )
                        self.agac.add_edge(prev_node, self.node_counter)
                        self.agac.add_edge(new_node, self.node_counter)
                        self.node_counter += 1
                        break

                self.active_nodes = temp_active_nodes

    def test(self, input_feature, verbose=False):
        new_node = 0
        # reset from input
        if input_feature < 0:
            if len(self.active_nodes) > 0:
                if self.agac.nodes[self.active_nodes[-1]]["depth"] > 3:
                    nei = list(self.agac.neighbors(self.active_nodes[-1]))
                    if len(nei) <= 1:
                        self.active_nodes = self.active_nodes + nei
                    else:
                        temp_active_nodes = []
                        for n in nei:
                            neii = list(self.agac.neighbors(n))
                            temp_active_nodes = temp_active_nodes + neii
                        self.active_nodes = temp_active_nodes
                for a_n in self.active_nodes:
                    if self.agac.out_degree(a_n) == 0:
                        if verbose:
                            print("trigger", a_n,
                                  nx.shortest_path(self.agac, source=0, target=a_n))  # self.agac.nodes[a_n]
                        self.active_nodes = []
                        return a_n
                return -1

            else:
                self.active_nodes = []
                return -1
        else:
            # get children of the root

            nei = list(self.agac.neighbors(0))
            feature_node = [n for n in nei if self.agac.nodes[n]['value'] == input_feature]

            # if the feature is available
            if len(feature_node) > 0:
                f_node = feature_node[0]
                # if not mature .. grow
                if self.agac.nodes[f_node]["occurrence_count"] < self.growth_threshold:
                    self.agac.nodes[f_node]["occurrence_count"] += 1

                # if mature
                else:
                    # f_node = feature_node[0]
                    self.agac.nodes[f_node]["occurrence_count"] += 1

                    new_node = f_node

            temp_active_nodes = []
            if new_node > 0:
                temp_active_nodes.append(new_node)

                for l, prev_node in enumerate(self.active_nodes):
                    if new_node == prev_node:
                        temp_active_nodes = self.active_nodes
                        break
                    if verbose:
                        print("layer", l, input_feature, prev_node, new_node)
                        # f_node = new_node

                    nei_c = list(self.agac.neighbors(new_node))
                    nei_p = list(self.agac.neighbors(prev_node))
                    common_node = list(set(nei_c) & set(nei_p))

                    if len(common_node) > 0:
                        self.agac.nodes[common_node[0]]["occurrence_count"] += 1

                        if self.agac.nodes[common_node[0]]["occurrence_count"] > self.growth_threshold:
                            if verbose:
                                print(len(self.active_nodes))
                            temp_active_nodes.append(common_node[0])
                            new_node = common_node[0]
                        else:
                            break
                    elif prev_node > 0:
                        break

                if len(temp_active_nodes) > 0 and len(self.active_nodes) > 0:
                    if self.agac.nodes[self.active_nodes[-1]]["depth"] - self.agac.nodes[temp_active_nodes[-1]][
                        "depth"] >= 2:
                        nei = list(self.agac.neighbors(self.active_nodes[-1]))
                        if len(nei) <= 1:
                            self.active_nodes = temp_active_nodes + nei
                        else:
                            for n in nei:
                                neii = list(self.agac.neighbors(n))
                                temp_active_nodes = temp_active_nodes + neii
                            self.active_nodes = temp_active_nodes
                    else:
                        self.active_nodes = temp_active_nodes
                else:
                    self.active_nodes = temp_active_nodes

                # self.active_nodes=temp_active_nodes
                for a_n in self.active_nodes:
                    if self.agac.out_degree(a_n) == 0:
                        if verbose:
                            print("trigger", a_n,
                                  nx.shortest_path(self.agac, source=0, target=a_n))  # self.agac.nodes[a_n]
                        self.active_nodes = []
                        return a_n
                return -1
            else:
                return -1

    def __init__(self, learn_mode=True, path="../../data/models/simple_symbol.pkl"):
        self.age = 0
        self.path = path
        self.learn_mode = learn_mode

        self.agac = nx.DiGraph()
        self.agac.add_node(
            0, value=-1, occurrence_count=10, id=0, depth=0
        )

        self.node_counter = 1

        # self.prev_index = -1
        # self.prev_input_feature = -10
        self.active_nodes = []

        self.growth_threshold = 3
        self.active_threshold = 3
