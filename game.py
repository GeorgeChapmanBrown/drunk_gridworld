import pygame
from sprites import *
from config import *
import sys

class Game:
	def __init__(self):
		pygame.init()
		self.screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
		self.clock = pygame.time.Clock()
		self.running = True

		self.character_spritesheet = Spritesheet("img/character.png")
		self.terrain_spritesheet = Spritesheet("img/terrain.png")
		self.bar_spritesheet = Spritesheet("img/pokemon.gif")
		self.house_image = Spritesheet("img/home.png")

	def createTileMap(self):
		for i , row in enumerate(tilemap):
			for j, column in enumerate(row):
				Ground(self, j, i)
				if column == 'D':
					Bar(self, j, i)
				if column == 'W':
					Pond(self, j, i)
				if column == 'B':
					Block(self, j, i)
				if column == 'P':
					Player(self, j, i)
				if column == 'H':
					Home(self, j, i)

	def new(self):
		self.playing = True

		self.all_sprites = pygame.sprite.LayeredUpdates()
		self.blocks = pygame.sprite.LayeredUpdates()

		self.createTileMap()

	def events(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.playing = False
				self.running = False

	def update(self, action):
		self.all_sprites.update(action)

	def draw(self):
		self.screen.fill(BLACK)
		self.all_sprites.draw(self.screen)
		self.clock.tick(FPS)
		pygame.display.update()

	def main(self, action):
		while self.playing:
			self.events()
			self.update(action)
			self.draw()
		self.running = False

	def introScreen(self):
		pass

def start_game(action):
	print('hi')
	g = Game()
	g.introScreen()
	g.new()
	while g.running:
		g.main(action)

	pygame.quit()
	sys.exit()
