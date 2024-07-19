import pygame as pg
import numpy as np
from numba import njit


world_map = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
])

def main():
    pg.init()
    screen = pg.display.set_mode((1270, 720))
    pg.mouse.set_visible(False)
    pg.event.set_grab(True)
    running = True
    clock = pg.time.Clock()
    
    hres = 120
    halfvres = 100
    
    mod = hres / 60
    posx, posy, rot = 1.5, 1.5, 0
    frame = np.random.uniform(0, 1, (hres, halfvres * 2, 3))
    
    wall_texture = pg.surfarray.array3d(pg.image.load('Texture/wall.png')) / 255.0
    
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT or event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                running = False
            elif event.type == pg.MOUSEMOTION:
                rot += event.rel[0] * 0.001
                
        frame = new_frame(posx, posy, rot, frame, hres, halfvres, mod, wall_texture)
        surf = pg.surfarray.make_surface(frame * 255)
        surf = pg.transform.scale(surf, (1270, 720))
        fps = int(clock.get_fps())
        pg.display.set_caption("DARK" + str(fps))
        
        screen.blit(surf, (0, 0))
        pg.display.update()
        
        posx, posy, rot = movement(posx, posy, rot, pg.key.get_pressed(), clock.tick())
        

def movement(posx, posy, rot, keys, et):
    if keys[pg.K_LEFT] or keys[ord('a')]:
        rot -= 0.001 * et
        
    if keys[pg.K_RIGHT] or keys[ord('d')]:
        rot += 0.001 * et
        
    if keys[pg.K_UP] or keys[ord('w')]:
        new_x, new_y = posx + np.cos(rot) * 0.002 * et, posy + np.sin(rot) * 0.002 * et
        if not is_collision(new_x, new_y):
            posx, posy = new_x, new_y
            
    if keys[pg.K_DOWN] or keys[ord('s')]:
        new_x, new_y = posx - np.cos(rot) * 0.002 * et, posy - np.sin(rot) * 0.002 * et
        if not  is_collision(new_x, new_y):
            posx, posy = new_x, new_y
            
    
    return posx, posy, rot


def is_collision(x, y):
    map_x, map_y = int(x), int(y)
    if 0 <= map_x < len(world_map[0]) and 0 <= map_y < len(world_map):
        return world_map[map_y][map_x] == 1
    return False


@njit()
def bilinear_interpolation(texture, x, y):
    x0, x1 = int(x), min(int(x) + 1, texture.shape[0] - 1)
    y0, y1 = int(y), min(int(y) + 1, texture.shape[1] - 1) 
    
    q11 = texture[x0, y0]
    q21 = texture[x1, y0]
    q12 = texture[x0, y1]
    q22 = texture[x1, y1]
    
    r1 = (x1 - x) * q11 + (x - x0) * q21
    r2 = (x1 - x) * q12 + (x - x0) * q22
    
    return (y1 - y) * r1 + (y - y0) * r2

@njit()
def new_frame(posx, posy, rot, frame, hres, halfvres, mod, wall_texture):
    for i in range(hres):
        rot_i = rot + np.deg2rad(i / mod - 30)
        sin, cos = np.sin(rot_i), np.cos(rot_i)
        cos2 = np.cos(np.deg2rad(i / mod - 30))
        
        sky_color = np.array([0.5, 0.7, 1.0])
        if i < frame.shape[0]:
            frame[i][:halfvres] = sky_color
        
        for j in range(halfvres):
            n = (halfvres / (halfvres - j)) / cos2
            x, y = posx + cos * n, posy + sin * n
            
            floor_color = np.array([0.3, 0.3, 0.3])
            shade = 0.2 + 0.8 * (1 - j / halfvres)
            if i < frame.shape[0] and halfvres + j < frame.shape[1]:
                frame[i][halfvres * 2 - j - 1] = shade * floor_color
            
            map_x, map_y = int(x), int(y)
            if 0 <= map_x < len(world_map[0]) and 0 <= map_y < len(world_map) and world_map[map_y][map_x] == 1:
                wall_height = int(halfvres / (n * cos2))
                wall_start = max(0, halfvres - wall_height)
                wall_end = min(halfvres * 2, halfvres + wall_height)
                
                if abs(x - map_x) > abs(y - map_y):
                    tex_x = y - map_y
                else:
                    tex_x = x - map_x
                
                tex_x *= (wall_texture.shape[0] - 1)
                     
                for k in range(wall_start, wall_end):
                    tex_y = (k - wall_start) * (wall_texture.shape[1] - 1) / (wall_end - wall_start)
                    if i < frame.shape[0] and k < frame.shape[1]:
                        frame[i][k] = bilinear_interpolation(wall_texture, tex_x, tex_y)
            
                break
                     
    return frame




if __name__ == '__main__':
    main()
    pg.quit()

            
    
    
    
    
    