
import random
import math
from numpy import exp
import utils


class DenseNetwork:
    def __init__(self, animal):
        self.layers = animal.weights_layers
        self.direc_ls = animal.coord_changes
        self.weights = [[[random.uniform(-1, 1) for weight in range(self.layers[l_idx + 1])] for node in range(self.layers[l_idx])] for l_idx in range(len(self.layers) - 1)]
        self.output = 0
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
        # return example [0, 0, 0, 1]
        # create the amount of outputs for up, down, left, right
        sliding_layer = inputs
        for layer_idx in range(len(self.weights)):
            layer_2 = self.weights[layer_idx]
            # print(f"layer2: {layer_2}")
            out_layer = [0] * len(layer_2[0])
            # print(f"out_layer: {out_layer}")
            for n_i in range(len(sliding_layer)):
                n = sliding_layer[n_i]
                for w_i in range(len(out_layer)):
                    for w in layer_2:
                        w = w[w_i]
                        out_layer[w_i] += n * w
            sliding_layer = out_layer
            # print(f"sliding_layer: {sliding_layer}")
            
        # print(self.softmax(sliding_layer))
        return self.softmax(sliding_layer)

    def adjust_coweightsections(self):
        alpha = self.sigmoid(self.cost) * .0001
        loc_min_update, loc_max_update = self.min_update + alpha, self.max_update + alpha
        # print(loc_min_update, loc_max_update)
        self.weights = [[[weight * random.uniform(loc_min_update, loc_max_update) for weight in node] for node in layer] for layer in self.weights]

    def map_input(self, curr_coord, focused_obj_coords, priority, health_diff):
        x, y = curr_coord[0], curr_coord[1]
        x2, y2 = focused_obj_coords[0], focused_obj_coords[1]
        dist = utils.distance_formula(x, y, focused_obj_coords[0], focused_obj_coords[1])
        self.cost = health_diff
        sign = -1 if priority == "predator" else 1
        return [dist, self.cost, sign]
    
    def think(self, curr_coord, focused_obj_coords, priority, health_diff):
        mapped_input = self.map_input(curr_coord, focused_obj_coords, priority, health_diff)
        # inputs = [self.distance_formula(self.focused_obj.x, self.focused_obj.y), self.focused_obj.x, self.focused_obj.y, self.x, self.y]
        outputs = list(self.propagate(mapped_input))
        self.output = outputs.index(max(outputs))
        self.adjust_coweightsections()
        # print(self.weights)
        # might need multiple weightss for multiple outputs?
        return self.output