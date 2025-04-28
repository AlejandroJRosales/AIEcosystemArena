from math import exp, sqrt


def distance_formula(x1, y1, x2, y2, mw=None, mh=None):
	if mw is not None and mh is not None:
		x1 = x1 % mw
		y1 = y1 % mh
		x2 = x2 % mw
		y2 = y2 % mh
		return sqrt(((x2 - x1) % mw)**2 + ((y2 - y1) % mh)**2)
	return sqrt(((x2 - x1)**2) + ((y2 - y1)**2))


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
	distance = distance_formula(x, y, x2, y2)
	if distance <= allowed:
		return True
	return False
