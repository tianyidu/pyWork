import pygame
from pygame.locals import *
from sys import exit

SCREEN_WIDTH = 390
SCREEN_HEIGHT = 520

pygame.init()
screen = pygame.display.set_mode([SCREEN_WIDTH,SCREEN_HEIGHT])
pygame.display.set_caption("Plane Game")

background = pygame.image.load("background.png")

while True:
	screen.blit(background,(0,0))
	pygame.display.update()
	for event in pygame.event.get():
		if event.type==pygame.QUIT:
			pygame.quit()
			exit()