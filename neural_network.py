
import random
import math
from numpy import exp


class DenseNetwork:
    def __init__(self, animal):
        self.layers = animal.nn_layers
        self.direc_ls = animal.coord_changes
        self.nn = [[[random.uniform(-1, 1) for weight in range(self.layers[l_idx + 1])] for node in range(self.layers[l_idx])] for l_idx in range(len(self.layers) - 1)]
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
        # print(inputs)
        # return [0, 0, 0, 1]
        # create the amount of outputs for up, down, left, right
        outputs = [[0 for node in range(layer_size)] for layer_size in self.layers]
        for l_idx in range(len(self.nn)):
            for n_idx in range(len(self.nn[l_idx])):
                for w_index in range(len(self.nn[l_idx][n_idx])):
                    if l_idx == 0:
                        outputs[l_idx + 1][w_index] += inputs[n_idx] * self.nn[l_idx][n_idx][w_index]
                    else:
                        outputs[l_idx + 1][w_index] += outputs[l_idx][n_idx] * self.nn[l_idx][n_idx][w_index]
        return self.softmax(outputs[-1])

    def adjust_connections(self):
        alpha = self.sigmoid(self.cost) * .0001
        loc_min_update, loc_max_update = self.min_update + alpha, self.max_update + alpha
        # print(loc_min_update, loc_max_update)
        self.nn = [[[weight * random.uniform(loc_min_update, loc_max_update) for weight in node] for node in layer] for layer in self.nn]

    def think(self, curr_coord, obj_locations, health_diff):
        # print(obj_locations)
        obj_locations = obj_locations if obj_locations is not None else (0, 0)
        x, y = curr_coord[0], curr_coord[1]
        x2, y2 = obj_locations[0], obj_locations[1]
        self.cost = health_diff
        inputs = [x,
                  y,
                  x2,
                  y2,
                  self.cost
                  ]
        # inputs = [self.distance_formula(self.focused_obj.x, self.focused_obj.y), self.focused_obj.x, self.focused_obj.y, self.x, self.y]
        outputs = list(self.propagate(inputs))
        self.output = outputs.index(max(outputs))
        self.adjust_connections()
        # print(self.nn)
        # might need multiple nns for multiple outputs?
        return self.output