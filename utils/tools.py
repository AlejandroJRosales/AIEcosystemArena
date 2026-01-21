from math import exp
import yaml
import os


def distance(x1, y1, x2, y2, mw=None, mh=None):
	if mw is not None and mh is not None:
		x1 = x1 % mw
		y1 = y1 % mh
		x2 = x2 % mw
		y2 = y2 % mh
		return ((x2 - x1) % mw)**2 + ((y2 - y1) % mh)**2
	return ((x2 - x1)**2) + ((y2 - y1)**2)


def clamp(value, minimum, maximum):
	return max((min(value, maximum)), minimum)


def logistic_curve(x):
	return 1 / (1 + exp(x))


def get_species_size(world, class_type):
	return len([obj for obj in world if isinstance(obj, class_type)])
	

def get_pop_sizes(world, species_types):
	pop_sizes = dict()
	for species in species_types.keys():
		pop_sizes[species] = get_species_size(world, species)
	return pop_sizes


def in_range(x, y, x2, y2, allowed):
	x = distance(x, y, x2, y2)
	if x <= allowed ** 2:
		return True
	return False

def resource_path(relative_path):
		""" Get absolute path to resource, works for dev and for PyInstaller """
		try:
			import sys
			# PyInstaller creates a temp folder and stores path in _MEIPASS
			base_path = sys._MEIPASS
		except Exception:
			base_path = os.path.abspath(".")
		return os.path.join(base_path, relative_path)


def extract_config_data(data_type):
	# user and default config paths
	user_config = os.path.join("mnt", f"{data_type}_config.yaml")
	default_config = os.path.join("assets", "config", f"default_{data_type}_config.yaml")
	
	# Check if user config exists, use default if not
	primary_asset_path = resource_path(user_config)
	if os.path.exists(primary_asset_path):
		asset_path = primary_asset_path
	else:
		asset_path = resource_path(default_config)
	
	with open(asset_path, "r") as file:
		data = yaml.safe_load(file)

	return data
