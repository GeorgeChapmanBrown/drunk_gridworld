import pygame
from config import *
import math
import random
import time
import sys

class Spritesheet:
	def __init__(self, file):
		self.sheet = pygame.image.load(file).convert()

	def get_sprite(self, x, y, width, height, colour):
		sprite = pygame.Surface([width, height])
		sprite.blit(self.sheet, (0,0), (x, y, width, height))
		if colour == 0:
			sprite.set_colorkey(BLACK)
		if colour == 1:
			sprite.set_colorkey(WHITE)
		return sprite

class Player(pygame.sprite.Sprite):
	def __init__(self, game, x, y):

		self.game = game
		self._layer = PLAYER_LAYER
		self.groups = self.game.all_sprites
		pygame.sprite.Sprite.__init__(self, self.groups)

		self.x = x * TILESIZE
		self.y = y * TILESIZE
		self.width = TILESIZE
		self.height = TILESIZE

		self.x_change = 0
		self.y_change = 0

		self.facing = 'down'

		self.image = self.game.character_spritesheet.get_sprite(3, 2, self.width, self.height, 0)

		self.rect = self.image.get_rect()
		self.rect.x = self.x
		self.rect.y = self.y

	def update(self, action):
		self.movement(action)

		self.rect.x += self.x_change
		self.collideBlocks('x')
		self.rect.y += self.y_change
		self.collideBlocks('y')

		self.pond()
		self.bar()
		self.home()

		self.x_change = 0
		self.y_change = 0

		# print(self.rect.x, self.rect.y)

	def movement(self, action):
		#add movement here
		# get from the generative model???
		# keys = pygame.key.get_pressed()

		# if keys[pygame.K_LEFT]:
		# 	for i in range(1):
		# 		self.x_change -= PLAYER_SPEED
		# 		self.facing = 'left'
		# 		time.sleep(0.1)

		# if keys[pygame.K_RIGHT]:
		# 	for i in range(1):
		# 		self.x_change += PLAYER_SPEED
		# 		self.facing = 'right' 
		# 		time.sleep(0.1)

		# if keys[pygame.K_UP]:
		# 	for i in range(1):
		# 		self.y_change -= PLAYER_SPEED
		# 		self.facing = 'up'
		# 		time.sleep(0.1)

		# if keys[pygame.K_DOWN]:
		# 	for i in range(1):
		# 		self.y_change += PLAYER_SPEED
		# 		self.facing = 'down'
		# 		time.sleep(0.1)  

		move = action.get()
		# move = 'right'

		if move == 'left':
			self.x_change -= PLAYER_SPEED
			self.facing = 'left'
			# time.sleep(0.1)

		elif move == 'right':
			self.x_change += PLAYER_SPEED
			self.facing = 'right'
			# time.sleep(0.1) 

		elif move == 'up':
			self.y_change -= PLAYER_SPEED
			self.facing = 'up'
			# time.sleep(0.1)

		elif move == 'down':
			self.y_change += PLAYER_SPEED
			self.facing = 'down'
			# time.sleep(0.1) 

		elif move == 'stay':
			self.x_change = 0
			self.y_change = 0
			self.facing = 'down' 
			# time.sleep(0.1)

	def collideBlocks(self, direction):
		if direction == "x":
			if self.rect.x < 64:
				self.rect.x = self.rect.x + PLAYER_SPEED
			if self.rect.x > 736:
				self.rect.x = self.rect.x - PLAYER_SPEED
		if direction == "y":
			if self.rect.y < 64:
				self.rect.y = self.rect.y + PLAYER_SPEED
			if self.rect.y > 256:
				self.rect.y = self.rect.y - PLAYER_SPEED

	def pond(self):
		self.pond_location = [[448, 64], [544, 64], [640, 64], [160, 256], [256, 256]]
		if [self.rect.x, self.rect.y] in self.pond_location:
			pass
			# print('\n\n\n\n\ndead\n\n\n\n\n')

	def bar(self):
		self.bar_location = [[160, 64], [352, 64], [448, 160], [640, 160]]
		if [self.rect.x, self.rect.y] in self.bar_location:
			pass
			# print('\n\n\n\n\ndrunk\n\n\n\n\n')

	def home(self):
		self.home_location = [[736, 256]]
		if [self.rect.x, self.rect.y] in self.home_location:
			# print('\n\n\n\n\nhome\n\n\n\n\n')
			pygame.quit()
			sys.exit()

class Block(pygame.sprite.Sprite):
	def __init__(self, game, x, y):

		self.game = game
		self._layer = BLOCK_LAYER
		self.groups = self.game.all_sprites, self.game.blocks
		pygame.sprite.Sprite.__init__(self, self.groups)

		self.x = x * TILESIZE
		self.y = y * TILESIZE
		self.width = TILESIZE
		self.height = TILESIZE

		self.image = self.game.terrain_spritesheet.get_sprite(352, 352, self.width, self.height, 0)

		self.rect = self.image.get_rect()
		self.rect.x = self.x
		self.rect.y = self.y

class Home(pygame.sprite.Sprite):
	def __init__(self, game, x, y):

		self.game = game
		self._layer = HOME_LAYER
		self.groups = self.game.all_sprites
		pygame.sprite.Sprite.__init__(self, self.groups)

		self.x = x * TILESIZE
		self.y = y * TILESIZE
		self.width = TILESIZE
		self.height = TILESIZE

		self.image = self.game.house_image.get_sprite(0, 18, 142, 200, 1)

		self.rect = self.image.get_rect()
		self.rect.x = self.x
		self.rect.y = self.y

class Bar(pygame.sprite.Sprite):
	def __init__(self, game, x, y):

		self.game = game
		self._layer = BAR_LAYER
		self.groups = self.game.all_sprites
		pygame.sprite.Sprite.__init__(self, self.groups)

		self.x = x * TILESIZE
		self.y = y * TILESIZE
		self.width = TILESIZE
		self.height = TILESIZE

		self.image = self.game.bar_spritesheet.get_sprite(146, 95, 97, 77, 0)

		self.rect = self.image.get_rect()
		self.rect.x = self.x
		self.rect.y = self.y

class Pond(pygame.sprite.Sprite):
	def __init__(self, game, x, y):

		self.game = game
		self._layer = POND_LAYER
		self.groups = self.game.all_sprites
		pygame.sprite.Sprite.__init__(self, self.groups)

		self.x = x * TILESIZE
		self.y = y * TILESIZE
		self.width = TILESIZE*3
		self.height = TILESIZE*3

		self.image = self.game.terrain_spritesheet.get_sprite(864, 64, self.width, self.height, 0)

		self.rect = self.image.get_rect()
		self.rect.x = self.x
		self.rect.y = self.y

class Ground(pygame.sprite.Sprite):
	def __init__(self, game, x, y):

		self.game = game
		self._layer = GROUND_LAYER
		self.groups = self.game.all_sprites
		pygame.sprite.Sprite.__init__(self, self.groups)

		self.x = x * TILESIZE
		self.y = y * TILESIZE
		self.width = TILESIZE
		self.height = TILESIZE

		self.image = self.game.terrain_spritesheet.get_sprite(64, 352, self.width, self.height, 0)

		self.rect = self.image.get_rect()
		self.rect.x = self.x
		self.rect.y = self.y