import numpy as np
import matplotlib.pyplot as plt
import species
import utils


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


def analysis_mode(obj):
	print(f"Object Type: {type(obj)}")
	print(f"Object ID: {id(obj)}")
	for k, v in obj.__dict__.items():
			print(f"{k}: {v}")
			print("-" * 36)
