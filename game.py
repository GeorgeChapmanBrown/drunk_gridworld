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
		self.font = pygame.font.Font('arial.ttf', 32)

		self.drunk_character = Spritesheet("img/drunk_spritesheet.png")
		self.drunk_character_down = Spritesheet("img/drunk_down.png")
		self.character_spritesheet = Spritesheet("img/character.png")
		self.terrain_spritesheet = Spritesheet("img/terrain.png")
		self.bar_spritesheet = Spritesheet("img/pokemon.gif")
		self.house_image = Spritesheet("img/home.png")
		self.go_background = pygame.image.load("img/gameover.png")

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

	def update(self, action, drunk):
		self.all_sprites.update(action, drunk)

	def draw(self):
		self.screen.fill(BLACK)
		self.all_sprites.draw(self.screen)
		self.clock.tick(FPS)
		pygame.display.update()

	def main(self, action, drunk):
		while self.playing:
			self.events()
			self.update(action, drunk)
			self.draw()

	def introScreen(self):
		pass

	def gameOver(self):
		text = self.font.render('Game Over', True, WHITE)
		text_rect = text.get_rect(center=(WIN_WIDTH/2, WIN_HEIGHT/2))

		exit_button = Button(10, WIN_HEIGHT - 60, 120, 50, WHITE, BLACK, 'EXIT', 32)

		for sprite in self.all_sprites:
			sprite.kill()

		while self.running:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					self.running = False

			mouse_pos = pygame.mouse.get_pos()
			mouse_pressed = pygame.mouse.get_pressed()

			if exit_button.is_pressed(mouse_pos, mouse_pressed):
				self.running = False

			self.screen.blit(self.go_background, (0,0))
			self.screen.blit(text, text_rect)
			self.screen.blit(exit_button.image, exit_button.rect)
			self.clock.tick(FPS)
			pygame.display.update()
		

def start_game(action, drunk):
	g = Game()
	g.introScreen()
	g.new()
	while g.running:
		g.main(action, drunk)

	pygame.quit()
	sys.exit()
