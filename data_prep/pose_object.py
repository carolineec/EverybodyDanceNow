class Pose:
	def __init__(self, keyname, posepts, facepts, rhandpts, lhandpts):
		self.keyname = keyname
		self.posepts = posepts
		self.facepts = facepts
		self.rhandpts = rhandpts
		self.lhandpts = lhandpts

	def update_lhand(self, newhand):
		self.lhandpts = newhand

	def update_rhand(self, newhand):
		self.rhandpts = newhand