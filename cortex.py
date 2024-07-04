from lib.feature import Feature
from lib.symbol import Symbol
from lib.expression import Expression
from lib.helper import *


class Cortex:
    def __init__(self):
        self.window_size = 128
        self.offset = 64
        self.path_to_store_data = "data/"

        parameters = {
            "path": "data/sensor_cortex.pkl",
            "learn_mode": True,
            "branch_depth": 16,
            "window_size": self.window_size,
        }
        self.feature = Feature(parameters)
        self.symbol = Symbol(path="data/symbol_cortex.pkl", learn_mode=True)
        self.expression = Expression(path="data/expression_cortex.pkl", learn_mode=True, forget_period=20)

    def _get_signal(self, data, dataset, test_set, train_test ):
        if train_test == "train":
            data_source = data[dataset]["train"+test_set]
        elif train_test=="test":
            data_source = data[dataset]["test"+test_set]
        else:
            data_source = data[dataset]["validation1"]
            signal = list(data_source["unsupervised_signal"])
            return signal

        signal = []
        for counter in range(len(data_source)):
            signal = signal + list(data_source[counter]["unsupervised_signal"])
        return signal

    
    def _train_feature(self, data, dataset, test_set):
        dissipated_energies_train = []
        past = [1000] * 1000
        av = []
        train_epochs=10
        signal = self._get_signal(data, dataset, test_set, train_test="train")
        signal_len = int((len(signal) - self.window_size) / self.offset + 1)

        for i in range(train_epochs):            

            

            for j in range(signal_len):
                nod, err = self.feature.step(signal[j * self.offset: j * self.offset + self.window_size])
                dissipated_energies_train.append(err)

                if nod > 0:
                    past.pop()
                    past.insert(0, err)
                    av.append(np.average(past))
                    if np.average(past) < 0.3:
                        break
            if np.average(past) < 0.3:            
                break


    def _train_symbol(self, data, dataset, test_set):
        symbol_active_states = []
        prev_node_count = 0
        train_epochs=10
        
        signal = self._get_signal(data, dataset, test_set, train_test="train")
        signal_len = int((len(signal) - self.window_size) / self.offset + 1)

        for i in range(train_epochs):         
            
            symbol_active_states.append([])
            for j in range(signal_len):
                feature_id, err = self.feature.step(signal[j * self.offset: j * self.offset + self.window_size])
                s = self.symbol.step(feature_id)
                if len(self.symbol.active_nodes) > 0:
                    symbol_active_states[-1].append(len(self.symbol.active_nodes))
            if prev_node_count == self.symbol.node_counter:
                break                
            else:
                prev_node_count = self.symbol.node_counter    

    def _train_expression(self, data, dataset, test_set):
        expression_active_states = []
        node_counts = []
        prev_node_count = 0
        train_epochs=10
        signal = self._get_signal(data, dataset, test_set, train_test="train")
        signal_len = int((len(signal) - self.window_size) / self.offset + 1)

        for i in range(train_epochs):     
            expression_active_states.append([])
            for j in range(signal_len):
                feature_id, err = self.feature.step(signal[j * self.offset: j * self.offset + self.window_size])
                symbol_id = self.symbol.step(feature_id)
                self.expression.step(symbol_id)
                if len(self.expression.aktive_nodes) > 1:
                    expression_active_states[-1].append(len(self.expression.aktive_nodes))
            node_counts.append(self.expression.counter)
            if len(node_counts) >= 2:
                if node_counts[-2] == node_counts[-1]:
                    break
    

    def start_training(self, dataset_path="data/dataset1.pkl", dataset="dataset_1", test_set = ""):
        print("training")
        data = pickle.load(open(dataset_path, "rb"))
        
        # self.feature = restore_object("data/sensor_cortex.pkl")
        # self.symbol = restore_object("data/symbol_cortex.pkl")
        self._train_feature(data,dataset, test_set)
        self.feature.learn_mode = False
        self._train_symbol(data,dataset, test_set)
        self.symbol.learn_mode = False
        self._train_expression(data,dataset, test_set)
        self.expression.learn_mode = False

    
    def predict (self, dataset_path="data/dataset1.pkl", dataset="dataset_1", test_set=""):
        data = pickle.load(open(dataset_path, "rb"))
        signal = self._get_signal(data, "validation", "", train_test="validation1")        
        signal_len = int((len(signal) - self.window_size) / self.offset + 1)

        equivalences=[]
        
        for j in range(signal_len):
            feature_id, err = self.feature.step(signal[j * self.offset: j * self.offset + self.window_size])
            symbol_id = self.symbol.step(feature_id)
            if symbol_id > 0:
                equivalences.append(symbol_id)                
                
        
    
        data_source = data[dataset]["test"+test_set]
        hits = 0

        for counter in range(len(data_source)):
            self.expression.aktive_nodes = [0]
            self.expression.input_buffer=[]
            
            signal = data_source[counter]["unsupervised_signal"]
            signal_len = int((len(signal) - self.window_size) / self.offset + 1)
            
            expected_value = data_source[counter]["label"]    
            expected_value = equivalences[expected_value]            

            prev_prev_pred = []
            prev_pred = []            
            for j in range(signal_len):
                feature_id, err = self.feature.step(signal[j * self.offset: j * self.offset + self.window_size])

                symbol_id = self.symbol.step(feature_id)        
                expression_id = self.expression.step(symbol_id)
                if symbol_id > 0:
                    prev_prev_pred = prev_pred
                    prev_pred = expression_id
            if prev_prev_pred:
                if expected_value == prev_prev_pred[0]:
                    hits+=1
        
        print("Expressions", len(data_source))
        print("predicted last symbol", hits)
        if len(data_source)>0:
            print("Accuracy",hits/len(data_source))
        else:
            print("Accuracy\t NA")
        
    
    
    
    