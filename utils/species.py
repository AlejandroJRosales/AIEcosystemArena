import os
import random
import math
import time
import numpy as np
import pygame
from pathlib import Path
from utils import tools
from utils import neural_network as ann
from utils.environment import Water
from utils.social import Predator


class SpeciesInfo:
	def __init__(self):
		self.data = self._process_config(tools.extract_config_data("species"))

	def _process_config(self, raw_data):
		name_to_class = {
			"Deer": Deer,
			"Wolf": Wolf,
			"Plant": Plant,
			"None": None
		}

		for species, attributes in raw_data.items():
			# Replace class name strings with actual class predatorects
			attributes["mate"] = name_to_class.get(attributes.get("mate"))
			if "predators" in attributes:
				attributes["predators"] = tuple(name_to_class.get(pred) for pred in attributes["predators"])
			if "diet" in attributes:
				attributes["diet"] = tuple(name_to_class.get(food) for food in attributes["diet"])

			# Evaluate random expressions safely
			for key, value in attributes.items():
				if isinstance(value, str) and value.startswith("random."):
					attributes[key] = eval(value)

		return raw_data

class Living(pygame.sprite.Sprite):
	def __init__(self, world, coord):
		super().__init__()
		self.world = world
		self.birth_coord = coord
		self.tob = time.time()
		image_path = tools.resource_path(os.path.join("assets", "images"))
		self.assets_img_path = Path(image_path)
		self.generation = 0

	def generate_entity(self, species_type, sexes_info):
		self.x = self.birth_coord[0]
		self.y = self.birth_coord[1]
		self.sex = random.choice(list(sexes_info.keys()))
		self.width = sexes_info[self.sex]["size"][0] * self.world.proportion
		self.height = sexes_info[self.sex]["size"][1] * self.world.proportion

		img_file_name = self.assets_img_path / f"{self.sex}-{species_type}.png"
		self.img = pygame.image.load(img_file_name)
		self.img = pygame.transform.scale(self.img, (self.width, self.height))
		self.rect = self.img.get_rect(center=(self.x, self.y))

	def get_coords(self):
		return self.x, self.y
		
	def attacked(self, predator):
		# TODO: tune weight of health depletion for animal and plant
		health_depl = self.start_health * .1
		# subtract health from self since being attacked
		self.health -= health_depl
		self.check()
		if isinstance(self, Animal) and isinstance(predator, Animal) and not self.alive:
			# if prey or self dead, add to health of predator or predator since animal is dead
			# print(f"{predator.species_type} {predator.x}, {predator.y} attacked {self} {self.x}, {self.y}")
			predator.health += self.start_health
			predator.food_need -= self.start_health
		if isinstance(self, Plant) and isinstance(predator, Animal):
			predator.health += health_depl
			predator.food_need -= health_depl

	def check(self):
		if self.health <= 1:
			self.die()

	def die(self):
		self.alive = False


class Animal(Living):
	def __init__(self, world, coord, species_info, species_type):
		super(Animal, self).__init__(world, coord)
		self.world = world
		self.species_type = species_type
		self.species_info = species_info
		self.sexes_info = self.species_info["sexes"]
		self.diet = self.species_info["diet"]
		self.predators = self.species_info["predators"]
		self.mate_pref = self.species_info["mate"]
		self.speed = self.species_info["speed"]
		self.consumption_rate = self.species_info["consumption_rate"]
		self.start_health = self.species_info["start_health"]
		self.generate_entity(self.species_type, self.sexes_info)
		self.memory = {
			"predator": None,
			"food": None,
			"water": None,
			"mate": None
		}
		self.priority_dict = {
			"predator": None,
			"food": None,
			"water": None,
			"mate": None
		}
		self.priority = None
		self.health = self.start_health
		self.age_depl = self.start_health * 0.00002
		self.water_depl = self.start_health * 0.001
		self.food_depl = self.start_health * 0.001
		self.water_need = round(random.uniform(0.01, 0.03), 4)
		self.food_need = round(random.uniform(0.01, 0.03), 4)
		self.reproduction_need = round(random.uniform(0.01, 0.03), 4)
		self.avoid_need = 0
		self.food_increment = self.health * 0.0125
		self.water_increment = self.health * 0.0001
		self.reproduction_increment = self.health * 0.0005
		# self.predator_reaction = 2
		self.last_child_tob = time.time()
		self.child_grace_period = random.randint(15, 30)
		self.look_for_mate = False
		self.vision_dist = random.randint(15, 100) * self.world.proportion
		self.mutation_multi = 0.5
		self.prob_mutation = 0.1
		self.alive = True
		self.is_focused = False
		self.is_exploring = False
		# self.focused_predator = None
		self.coords_focused = Coords()
		self.is_player = False

		# brain
		self.coord_changes = [(self.speed, 0), (-self.speed, 0), (0, self.speed), (0, -self.speed), (self.speed, self.speed), (self.speed, -self.speed), (-self.speed, self.speed), (-self.speed, -self.speed)]
		self.num_inputs = 6 # TODO: map dynamic input dim
		self.nn_layer_dims = [self.num_inputs, self.num_inputs, 9, len(self.coord_changes), len(self.coord_changes)]
		self.brain = random.choice([ann.DenseNetwork, ann.LSTMNetwork])(self)
		self.output = None

	def neighbors(self, predators):
		return [predator for predator in predators if tools.distance(self.x, self.y, predator.x, predator.y) <= self.vision_dist and id(self) != id(predator)]

	def normalize_direction_focused(self):
		v = np.subtract((self.coords_focused.x, self.coords_focused.y), (self.x, self.y))
		normalized_v = v / np.linalg.norm(v)
		is_predator = any([issubclass(food, Animal) for food in self.diet])
		if self.priority == "predator" or (is_predator and self.priority == "food"):
			return normalized_v * self.speed
		return normalized_v * (self.speed * 0.3)

	def move(self, coord_idx=None):
		coord_change = self.coord_changes[coord_idx]
		self.x = (coord_change[0] + self.x) % self.world.world_width
		self.y = (coord_change[1] + self.y) % self.world.world_height
		self.rect.center = (self.x, self.y)

	def focus(self, predator, type):
		self.memory[type] = (predator.x, predator.y)
		self.is_focused = True

	def transpose_focused_coords(self):
		self.coords_focused.x, self.coords_focused.y = self.x + (
			self.x - self.coords_focused.x), self.y + (self.y - self.coords_focused.y)
			
	def calc_avoid_needed(self):
		# make it so that if a predator is after self then make that take precedence
		self.avoid_need = math.inf

	def search(self, predators):
		predator_distances = {
			"predator": math.inf,
			"food": math.inf,
			"water": math.inf,
			"mate": math.inf
		}
		
		# neighboring_predators = [predator for predator in predators if tools.distance(self.x, self.y, predator.x, predator.y) <= self.vision_dist * 0.3]
		for predator in predators:
			predator_dist = tools.distance(self.x, self.y, predator.x, predator.y)
			
			"""
			wx = self.world.x
			wy = self.world.y
			x_diff = self.x + predator.x - wx
			y_diff = self.y + predator.y - wy
			in_sight = None
			if x_diff > 0:
				in_sight = x_diff <= self.vision_dist
			elif y_diff > 0:
				in_sight = y_diff <= self.vision_dist
			else:
				in_sight = predator_dist <= self.vision_dist
			"""
			
			in_sight = predator_dist <= self.vision_dist
				
			if in_sight:
				# first condition checks if self has predators to prevent runtime error
				# second condition checks if predator is a predator of self
				predator_is_predator = self.predators[0] is not None and isinstance(predator, self.predators)
				if predator_is_predator and predator_dist <= predator_distances["predator"] and id(predator) != id(self):
					predator_distances["predator"] = predator_dist
					self.focus(predator, "predator")
					# calc avoidance here because it will be used later
					self.calc_avoid_needed()
					# todo: calculate avoidance needed for each animal
				
				# if the predator is food
				elif isinstance(predator, self.diet) and predator_dist <= predator_distances["food"] and id(predator) != id(self):
					predator_distances["food"] = predator_dist
					self.focus(predator, "food")
					
				# if the predatorect is water
				elif isinstance(predator, Water) and predator_dist <= predator_distances["water"] and id(predator) != id(self):
					predator_distances["water"] = predator_dist
					self.focus(predator, "water")
				
				# if the predator is a mate
				elif isinstance(predator, self.mate_pref) and predator_dist <= predator_distances["mate"] and predator.sex != self.sex and id(predator) != id(self):
					predator_distances["mate"] = predator_dist
					self.focus(predator, "mate")

	def think(self, predators):
		if not self.is_player:
			self.search(predators)
			needs = {
				"predator": self.avoid_need,
				"food": self.food_need,
				"water": self.water_need,
				"mate": self.reproduction_need
			}

			# selfs priorities and respective need amount
			# old implementation for choosing next move
			priority_list = sorted(needs.items(), key=lambda item: item[1], reverse=True)
			self.priority_dict = dict(priority_list)
	

			# starts with first priority
			for need in self.priority_dict.keys():
				self.predator_location = self.memory[need]
				# if self knows location of priority and thus location not none go to coords
				if self.predator_location is not None:
					self.priority = need
					# self.coords_focused.x, self.coords_focused.y = predator_location[0], predator_location[1]
					# if predator focused on is a predator transpose coords to go the
					# opposite direction
					# if need == "predator":
					# 	self.transpose_focused_coords()
					# end here since a higher priority predator was found and located
					break

			# feed predator_locations, priorities, and difference in health as cost function through nn
			focused_predator_coords = self.predator_location
			if focused_predator_coords is not None:
				agents_choice = self.brain.think(self)
			else:
				# if no predator found then move randomly
				agents_choice = random.randint(0, len(self.coord_changes) - 1)
			# go to location
			self.output_idx = agents_choice
			self.move(self.output_idx)

	def update_resources_need(self):
		# self.water_need = tools.clamp(self.water_need + self.water_increment, 0, 1)
		# self.food_need = tools.clamp(self.food_need + self.food_increment, 0, 1)
		# if not looking for mate set reproduction to 0
		# self.reproduction_need = tools.clamp(self.reproduction_need + self.reproduction_increment, 0, 1) if self.look_for_mate else 0
		
		self.water_need = self.water_need + self.water_increment
		self.food_need = self.food_need + self.food_increment
		# if not looking for mate set reproduction to 0
		self.reproduction_need = self.reproduction_need + self.reproduction_increment if self.look_for_mate else 0
		
		# avoid_needed calculated in think

	def update_internal_clocks(self):
		self.look_for_mate = time.time() - self.last_child_tob >= self.child_grace_period
		
	def update_memory(self):
		# self.memory = {
		#	"predator": None,
		#	"food": None,
		#	"water": None,
		#	"mate": None
		# }
		# if the predatorect is farther than the animal can see
		for predator_key in self.memory.keys():
			predator_loc = self.memory[predator_key]
			if predator_loc is not None and tools.distance(self.x, self.y, predator_loc[0], predator_loc[0]) <= self.vision_dist:
				self.memory[predator_key] = None

	def update_body(self):
		self.is_exploring = not self.is_focused
		self.is_focused = False
		self.priority = None
		self.update_resources_need()
		self.update_internal_clocks()
		self.update_memory()
		if tools.in_range(self.x, self.y, self.coords_focused.x, self.coords_focused.y, 7) or random.random() <= 0.001:
			self.new_explore_coords()
		self.health -= (time.time() - self.tob) * self.age_depl
		self.health -= self.water_need * 0.0001
		self.health -= self.food_need * 0.0001
		self.avoid_need = 0

	def update(self, envir_class, predators):
		neighbors = self.neighbors(predators)
		self.update_body()
		self.think(neighbors)
		self.detect_collision(envir_class, neighbors)
		self.check()
		
	def new_explore_coords(self):
		self.coords_focused.x = self.world.world_width * random.uniform(0, 1)
		self.coords_focused.y = self.world.world_height * random.uniform(0, 1)

	def detect_collision(self, envir_class, predators):
		for predator in predators:
			if self.rect.colliderect(predator.rect):
				# Avoid mating with self
				if self.look_for_mate and self.sex == "female" and isinstance(predator, self.mate_pref) and self.sex != predator.sex and id(self) != id(predator):
					envir_class.children.append(self.mate(predator))
					# print(f"{self.species_type} {self.x}, {self.y} mated with {predator.species_type} {predator.x}, {predator.y}")
					return  # Stop after mating

				# Attack prey
				if type(predator) in self.diet:
					predator.attacked(self)

				# Drink water
				elif isinstance(predator, Water):
					self.water_need -= self.water_increment

	def mate(self, parent2):
		child = type(parent2)(self.world, (self.x, self.y - (self.width + 5)))
		lower_bound = 1 - self.mutation_multi if 1 - self.mutation_multi >= 0.1 else 0.1
		upper_bound = 1 + self.mutation_multi if 1 + self.mutation_multi <= 4 else 4
		
		# take the average of the parents attributes and mutate them
		# mutate_attr = [(self.speed, child.speed), (self.vision_dist, child.vision_dist)]
		# for attr in mutate_attr:
		#     attr[0] = ((self.speed + parent2.speed) / 2) * random.uniform(lower_bound, upper_bound)
		#     child.vision_dist = ((self.vision_dist + parent2.vision_dist) // 2) * random.uniform(lower_bound,
		#                                                                                          upper_bound)
		# child.coord_changes = [(0, child.speed), (0, -child.speed), (child.speed, 0), (-child.speed, 0)]
		
		child.speed = ((self.speed + parent2.speed) / 2) * random.uniform(
			lower_bound,
			upper_bound)
		child.coord_changes = [(child.speed, 0), (-child.speed, 0), (0, child.speed), (0, -child.speed), (child.speed, child.speed), (child.speed, -child.speed), (-child.speed, child.speed), (-child.speed, -child.speed)]
		child.vision_dist = (
			(self.vision_dist + parent2.vision_dist) / 2) * random.uniform(
				lower_bound, upper_bound
			)
		child.mutation_multi = ((self.mutation_multi + parent2.mutation_multi) / 2) * random.uniform(0.9, 1.1)
		# build brain based off of parents brains, then reinitialaize nn
		child.brain = random.choice([type(self.brain), type(parent2.brain)])(child)
		child.brain.learning_rate = ((self.brain.learning_rate + parent2.brain.learning_rate) / 2) * random.uniform(0.5, 1.5)
		
		# learned experiences dont get inherited
		# child.brain.crossover(self.brain, parent2.brain, mutation_multi=self.mutation_multi)

		self.last_child_tob = time.time()
		self.update_internal_clocks()
		self.reproduction_need = 0
		self.generation = max(self.generation, parent2.generation) + 1

		parent2.last_child_tob = time.time()
		parent2.update_internal_clocks()
		parent2.reproduction_need = 0

		return child


class Deer(Animal):
	def __init__(self, world, coord):
		self.species_info = SpeciesInfo().data["deer"]
		super(Deer, self).__init__(world, coord, self.species_info, "deer")


class Wolf(Animal, Predator):
	def __init__(self, world, coord):
		self.species_info = SpeciesInfo().data["wolf"]
		super(Wolf, self).__init__(world, coord, self.species_info, "wolf")


class Plant(Living):
	def __init__(self, world, coord):
		super(Plant, self).__init__(world, coord)
		self.species_info = SpeciesInfo().data["plant"]
		self.sexes_info = self.species_info["sexes"]
		self.generate_entity("plant", self.sexes_info)
		self.health = 30
		self.start_health = self.health
		self.size_multi = 0.6
		self.alive = True

	def update(self):
		self.check()


class Coords:
	x = -1
	y = -1
