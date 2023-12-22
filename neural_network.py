
import random
from numpy import exp


class DenseNetwork:
    def __init__(self, animal):
        self.layers = animal.nn_layers
        self.direc_ls = animal.coord_changes
        self.nn = [[[random.uniform(-1, 1) for weight in range(self.layers[l_idx + 1])] for node in range(self.layers[l_idx])] for l_idx in range(len(self.layers) - 1)]
        self.output = 0
        
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

    def think(self, curr_coord, obj_locations, health_diff):
        # print(obj_locations)
        obj_locations = obj_locations if obj_locations is not None else (0, 0)
        x, y = curr_coord[0], curr_coord[1]
        x2, y2 = obj_locations[0], obj_locations[1]
        inputs = [x,
                  y,
                  x2,
                  y2,
                  health_diff
                  ]
        # inputs = [self.distance_formula(self.focused_obj.x, self.focused_obj.y), self.focused_obj.x, self.focused_obj.y, self.x, self.y]
        outputs = list(self.propagate(inputs))
        self.output = outputs.index(max(outputs))
        # might need multiple nns for multiple outputs?
        return self.output