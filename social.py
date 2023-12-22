class Predator:
	def set_pack(self, self_animal, objs):
		print("setting pack")
		pack = []
		for obj in objs:
			# if another animal of same species has the same prey in mind and they are
			# not yourself, then pack hunt
			# todo: only animals, currently accepts water and plants i think
			if self_animal.coords_focused == obj.coords_focused and self_animal.species_type == obj.species_type and id(self_animal) != id(obj):
				pack.append(obj)
		return pack
				
	def set_formation(self, pack):
		print("setting formation with", pack)
	
	def pack_hunt(self, self_animal, neighbors):
		pack = self.set_pack(self_animal, neighbors)
		self.set_formation(pack)
