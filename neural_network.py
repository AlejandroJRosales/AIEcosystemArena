
import random
import math
from numpy import exp
import utils


# neural evolution augmenting topologies
class NEAT:
    pass

class DenseNetwork:
    def __init__(self, animal):
        # TODO: tune hyperparameters
        self.layers = animal.weights_layers
        # print(self.layers)
        self.weights = [[[random.uniform(-1, 1) for weight in range(self.layers[l_idx + 1])] for node in range(self.layers[l_idx])] for l_idx in range(len(self.layers) - 1)]
        self.weights.append([[random.uniform(-1, 1) for weight in range(self.layers[-2])] for node in range(self.layers[-1])])
        # print([len(l) for l in self.weights])
        self.min_update = 0.9998
        self.max_update = 1.0002

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
        
    @staticmethod
    def softmax(vector):
        e = exp(vector)
        return e / e.sum()

    def propagate(self, inputs):
        # TODO: dynmaic input weight insertion, input weight generation and insertion code comented out for
        # self.layers.insert(0, len(inputs))
        # input_weights = [[random.uniform(-1, 1) for weight in range(self.layers[1])] for node in range(self.layers[0])]
        # self.weights.insert(0, input_weights)
        sliding_layer = inputs
        for l_idx in range(len(self.layers) - 1):
            out_layer = [0] * self.layers[l_idx + 1]
            for n1_idx in range(self.layers[l_idx]):
                for n2_idx in range(self.layers[l_idx + 1]):
                    current_node = sliding_layer[n1_idx]
                    weight = self.weights[l_idx][n1_idx][n2_idx]
                    out_layer[n2_idx] += current_node * weight
            sliding_layer = out_layer
        return self.softmax(sliding_layer)

    def adjust_weights(self):
        # TODO: implement heritable learning rate aka alpha value 
        alpha = self.sigmoid(self.cost) * .0001
        loc_min_update, loc_max_update = self.min_update + alpha, self.max_update + alpha
        # print(loc_min_update, loc_max_update)
        self.weights = [[[weight * random.uniform(loc_min_update, loc_max_update) for weight in node] for node in layer] for layer in self.weights]

    def map_input(self, curr_coord, focused_obj_coords, priority, health_diff):
        x, y = curr_coord[0], curr_coord[1]
        x2, y2 = focused_obj_coords[0], focused_obj_coords[1]
        # dist = utils.distance_formula(x, y, focused_obj_coords[0], focused_obj_coords[1])
        self.cost = health_diff
        sign = -1 if priority == "predator" else 1
        return [x, y, x2, y2, sign]
    
    def think(self, curr_coord, focused_obj_coords, priority, health_diff):
        mapped_input = self.map_input(curr_coord, focused_obj_coords, priority, health_diff)
        # inputs = [self.distance_formula(self.focused_obj.x, self.focused_obj.y), self.focused_obj.x, self.focused_obj.y, self.x, self.y]
        values = list(self.propagate(mapped_input))
        self.output = max(range(len(values)), key=values.__getitem__)
        self.adjust_weights()
        # print(self.weights)
        # might need multiple weightss for multiple outputs?
        return self.output