import random
import math
import time
import numpy as np
import utils
from environment import Water
from social import Predator
import neural_network as ann
import player
import pygame



class SpeciesInfo:
	def __init__(self):
		self.data = {
			"deer": {
				"sexes": {
					"male": {
						"size": (20, 30)
					},
					"female": {
						"size": (20, 30)
					}
				},
				"mate": Deer,
				"predators": (Wolf, ),
				"diet": (Plant, ),
				"speed": random.uniform(3, 5),
				"consumption rate": random.randint(2, 5)
			},
			"wolf": {
				"sexes": {
					"male": {
						"size": (20, 30)
					},
					"female": {
						"size": (20, 30)
					}
				},
				"predators": (None, ),
				"mate": Wolf,
				"diet": (Deer, ),
				"speed": random.uniform(3, 5),
				"consumption rate": random.randint(30, 60)
			},
			"plant": {
				"sexes": {
					"male": {
						"size": (8, 10)
					},
					"female": {
						"size": (8, 10)
					}
				},
				"mate": Plant,
			}
		}


class Living(pygame.sprite.Sprite):
	def __init__(self, world, coord):
		super(Living, self).__init__()
		pygame.sprite.Sprite.__init__(self)
		self.world = world
		self.birth_coord = coord

	def generate_entity(self, species_type, sexes_info):
		self.x = self.birth_coord[0]
		self.y = self.birth_coord[1]
		self.sex = random.choice(list(sexes_info.keys()))
		self.width = sexes_info[self.sex]["size"][0] * self.world.proportion
		self.height = sexes_info[self.sex]["size"][1] * self.world.proportion

		img_file_name = f"images/{self.sex}-{species_type}.png"
		self.img = pygame.image.load(img_file_name)
		self.img = pygame.transform.scale(self.img, (self.width, self.height))
		self.rect = pygame.rect.Rect((self.x, self.y, self.width, self.height))
		self.rect.center = (self.x, self.y)

	def get_coords(self):
		return self.x, self.y
		
	def attacked(self, obj):
		health_depl = self.start_health * .02
		
		# subtract health from self since being eaten
		self.health -= health_depl

		# add to health of obj since it is eating part of self
		obj.health += health_depl
		obj.food_need -= health_depl

		self.check()

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
		self.consumption_rate = self.species_info["consumption rate"]
		self.generate_entity(self.species_type, self.sexes_info)
		self.obj_locations = {
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
		self.start_health = random.randint(75, 150)
		self.health = self.start_health
		self.age_depl = self.start_health * 0.00002
		self.water_depl = self.start_health * 0.001
		self.food_depl = self.start_health * 0.001
		self.tob = time.time()
		self.water_need = round(random.uniform(0.01, 0.03), 4)
		self.food_need = round(random.uniform(0.01, 0.03), 4)
		self.reproduction_need = round(random.uniform(0.01, 0.03), 4)
		self.avoid_need = 0
		self.food_increment = self.health * 0.000125
		self.water_increment = self.health * 0.0001
		self.reproduction_increment = self.health * 0.0005
		# self.predator_reaction = 2
		self.last_child_tob = time.time()
		self.child_grace_period = random.randint(50, 60)
		self.look_for_mate = False
		self.vision_dist = random.randint(150, 300)
		self.mutation_multi = 0.2
		self.prob_mutation = 0.1
		self.alive = True
		self.is_focused = False
		self.is_exploring = False
		# self.focused_obj = None
		self.coords_focused = Coords()
		self.is_player = False

		# brain
		self.coord_changes = [(self.speed, 0), (-self.speed, 0), (0, self.speed), (0, -self.speed), (self.speed, self.speed), (self.speed, -self.speed), (-self.speed, self.speed), (-self.speed, -self.speed)]
		self.num_inputs = 5
		self.weights_layers = [self.num_inputs, 4, 3, 4, 3, len(self.coord_changes)]
		print(f"\t\t\t\tCreating neural network for {self.species_type}:{id(self)}...")
		self.brain = ann.DenseNetwork(self)

	def neighbors(self, objs):
		return [obj for obj in objs if utils.distance_formula(self.x, self.y, obj.x, obj.y) <= self.vision_dist and id(self) != id(obj)]

	def normalize_direction_focused(self):
		v = np.subtract((self.coords_focused.x, self.coords_focused.y), (self.x, self.y))
		normalized_v = v / np.linalg.norm(v)
		is_predator = any([issubclass(food, Animal) for food in self.diet])
		if self.priority == "predator" or (is_predator and self.priority == "food"):
			return normalized_v * self.speed
		return normalized_v * (self.speed * 0.3)

	def move(self, coord_idx=None):
		# AHHH I WAS NORMALIZING THE VECTORS WHEN IT SHOULD NOT BE DONE
		try:
			coord_change = self.coord_changes[coord_idx]
		except:
			print(coord_idx)
			assert 1 == 2
		self.x = (coord_change[0] + self.x) % self.world.world_width
		self.y = (coord_change[1] + self.y) % self.world.world_height
		self.rect.center = (self.x, self.y)

	def focus(self, obj, type):
		self.obj_locations[type] = (obj.x, obj.y)
		self.is_focused = True

	def transpose_focused_coords(self):
		self.coords_focused.x, self.coords_focused.y = self.x + (
			self.x - self.coords_focused.x), self.y + (self.y - self.coords_focused.y)
			
	def calc_avoid_needed(self):
		# make it so that if a predator is after self then make that take precedence
		self.avoid_need = math.inf

	def search(self, objs):
		obj_distances = {
			"predator": math.inf,
			"food": math.inf,
			"water": math.inf,
			"mate": math.inf
		}
		
		# neighboring_predators = [obj for obj in objs if utils.distance_formula(self.x, self.y, obj.x, obj.y) <= self.vision_dist * 0.3]
		for obj in objs:
			obj_dist = utils.distance_formula(self.x, self.y, obj.x, obj.y)
			
			"""
			wx = self.world.x
			wy = self.world.y
			x_diff = self.x + obj.x - wx
			y_diff = self.y + obj.y - wy
			in_sight = None
			if x_diff > 0:
				in_sight = x_diff <= self.vision_dist
			elif y_diff > 0:
				in_sight = y_diff <= self.vision_dist
			else:
				in_sight = obj_dist <= self.vision_dist
			"""
			
			in_sight = obj_dist <= self.vision_dist
				
			if in_sight:
				# first condition checks if self has predators to prevent runtime error
				# second condition checks if obj is a predator of self
				obj_is_predator = self.predators[0] is not None and isinstance(obj, self.predators)
				if obj_is_predator and obj_dist <= obj_distances["predator"] and id(obj) != id(self):
					obj_distances["predator"] = obj_dist
					self.focus(obj, "predator")
					# calc avoidance here because it will be used later
					self.calc_avoid_needed()
					# todo: calculate avoidance needed for each animal
				
				# if the obj is food
				elif isinstance(obj, self.diet) and obj_dist <= obj_distances["food"] and id(obj) != id(self):
					obj_distances["food"] = obj_dist
					self.focus(obj, "food")
					
				# if the object is water
				elif isinstance(obj, Water) and obj_dist <= obj_distances["water"] and id(obj) != id(self):
					obj_distances["water"] = obj_dist
					self.focus(obj, "water")
				
				# if the obj is a mate
				elif isinstance(obj, self.mate_pref) and obj_dist <= obj_distances["mate"] and obj.sex != self.sex and id(obj) != id(self):
					obj_distances["mate"] = obj_dist
					self.focus(obj, "mate")

	def think(self, objs):
		if not self.is_player:
			self.search(objs)
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
				self.obj_location = self.obj_locations[need]
				# if self knows location of priority and thus location not none go to coords
				if self.obj_location is not None:
					self.priority = need
					# self.coords_focused.x, self.coords_focused.y = obj_location[0], obj_location[1]
					# if obj focused on is a predator transpose coords to go the
					# opposite direction
					# if need == "predator":
					# 	self.transpose_focused_coords()
					# end here since a higher priority obj was found and located
					break

			# feed obj_locations, priorities, and difference in health as cost function through nn
			health_diff = self.health - self.start_health
			curr_coord = (self.x, self.y)
			focused_obj_coords = self.obj_location
			if focused_obj_coords is not None:
				agents_choice = self.brain.think(curr_coord, focused_obj_coords, self.priority, health_diff)
				# print(agents_choice)
				# go to location
				self.move(agents_choice)

	def update_resources_need(self):
		# self.water_need = utils.clamp(self.water_need + self.water_increment, 0, 1)
		# self.food_need = utils.clamp(self.food_need + self.food_increment, 0, 1)
		# if not looking for mate set reproduction to 0
		# self.reproduction_need = utils.clamp(self.reproduction_need + self.reproduction_increment, 0, 1) if self.look_for_mate else 0
		
		self.water_need = self.water_need + self.water_increment
		self.food_need = self.food_need + self.food_increment
		# if not looking for mate set reproduction to 0
		self.reproduction_need = self.reproduction_need + self.reproduction_increment if self.look_for_mate else 0
		
		# avoid_needed calculated in think

	def update_internal_clocks(self):
		self.look_for_mate = time.time(
		) - self.last_child_tob >= self.child_grace_period
		
	def update_memory(self):
		# self.obj_locations = {
		#	"predator": None,
		#	"food": None,
		#	"water": None,
		#	"mate": None
		# }
		# if the object is farther than the animal can see
		for obj_key in self.obj_locations.keys():
			obj_loc = self.obj_locations[obj_key]
			if obj_loc is not None and utils.distance_formula(self.x, self.y, obj_loc[0], obj_loc[0]) <= self.vision_dist:
				self.obj_locations[obj_key] = None

	def update_body(self):
		self.is_exploring = not self.is_focused
		self.is_focused = False
		self.priority = None
		self.update_resources_need()
		self.update_internal_clocks()
		self.update_memory()
		if utils.in_range(self.x, self.y, self.coords_focused.x, self.coords_focused.y, 7) or random.random() <= 0.001:
			self.new_explore_coords()
		self.health -= (time.time() - self.tob) * self.age_depl
		self.health -= self.water_need * 0.0001
		self.health -= self.food_need * 0.0001
		self.avoid_need = 0

	def update(self, envir_class, objs):
		neighbors = self.neighbors(objs)
		self.update_body()
		self.think(neighbors)
		self.detect_collision(envir_class, neighbors)
		self.check()
		
	def new_explore_coords(self):
		self.coords_focused.x = self.world.world_width * random.uniform(0, 1)
		self.coords_focused.y = self.world.world_height * random.uniform(0, 1)

	def detect_collision(self, envir_class, objs):
		for obj in objs:
			# if self.collide(obj):
			if self.rect.colliderect(obj):
				# make sure the entity is not attacking itself since itself is in the list of entities
				# a male and female will find each other, so all coniditons past the first one will be true  for male
				# and female. Meaning the male will have a kid and the female, we want just the choosen sex to give birth
				if self.sex == "female" and self.look_for_mate and isinstance(obj, self.mate_pref) and self.sex != obj.sex and id(self) != id(obj):
					# todo: look for best mate out of potential mates near by
					envir_class.children.append(self.mate(obj))
				elif type(obj) in self.diet:
					obj.attacked(self)
				elif isinstance(obj, Water):
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
			(self.vision_dist + parent2.vision_dist) // 2) * random.uniform(
				lower_bound, upper_bound
			)
		# child.mutation_multi = ((self.mutation_multi + parent2.mutation_multi) / 2) * random.uniform(0.9, 1.1)

		# TODO: need to crossover parent 1 init neural network state with parents 2 init neural network state, NOT current state

		self.last_child_tob = time.time()
		self.update_internal_clocks()
		self.reproduction_need = 0

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
