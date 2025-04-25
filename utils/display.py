import numpy as np
import matplotlib.pyplot as plt
from utils import species
from utils.neural_network import DenseNetwork


def text_pop_sizes(world):
	print("Deer: ", len([obj for obj in world if isinstance(obj, species.Deer)]))
	print("Wolfs: ", len([obj for obj in world if isinstance(obj, species.Wolf)]))
	print("Plants: ", len([obj for obj in world if isinstance(obj, species.Plant)]))
	print()
	# LabelNode.__init__(self, "test")
	
	
def chart_pop_sizes(sizes_time):
	print(sizes_time)
	exit()
	for species in len(sizes_time[0]):
		plt.fill_between(list(sizes_time.keys()), list(sizes_time.values()))
	plt.show()


def print_weights_tree(weights, indent="│   ", max_neurons=3, precision=4):
	layer_indent = indent + "│   "
	for i, layer in enumerate(weights):
		print(f"{layer_indent}├── Layer {i}")
		for j, neuron_weights in enumerate(layer):
			if j >= max_neurons:
				print(f"{layer_indent}│   ├── ...")
				break
			formatted = ", ".join(f"{w:.{precision}f}" for w in neuron_weights)
			print(f"{layer_indent}│   ├── Neuron {j}: [{formatted}]")


def analysis_mode(obj, indent=""):
	print(f"{indent}{type(obj).__name__} (id={id(obj)})")
	sub_indent = indent + "│   "

	for attr, value in obj.__dict__.items():
		if isinstance(value, DenseNetwork):
			print(f"{sub_indent}├── {attr}: DenseNetwork")
			dense_indent = sub_indent + "│   "
			for sub_attr, sub_value in value.__dict__.items():
				if sub_attr == "weights":
					print(f"{dense_indent}├── weights:")
					print_weights_tree(sub_value, dense_indent)
				else:
					print(f"{dense_indent}├── {sub_attr}: {sub_value}")
		else:
			print(f"{sub_indent}├── {attr}: {value}")

