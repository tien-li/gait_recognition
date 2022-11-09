import pygame as pg
import sys
# define colors (RGB)
black = 0, 0, 255
red = 255, 0, 0

# initializing the window
pg.init()
size = width, height = 640, 480
screen = pg.display.set_mode(size)

# bouncing ball
px,py = 320,240
vx,vy = 20,20
while 1:
    for event in pg.event.get():
        if event.type == pg.QUIT: sys.exit()
    px += vx
    py += vy
    if px < 0 or px > width:
        vx = -vx
    if py < 0 or py > height:
        vy = -vy
    screen.fill(black)
    pg.draw.circle(screen,red,(px,height-py),4,0)
    pg.display.flip()