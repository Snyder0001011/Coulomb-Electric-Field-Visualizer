import numpy as np
import matplotlib.pyplot as plt
import math
import pygame
import numpy as np

epsilon_0 = 8.854187817e-12 

pygame.init()
Width, Height  = 1920 , 1080
screen = pygame.display.set_mode((Width, Height))
pygame.display.set_caption("Electric Field Visualizer")

WHITE = (255, 255, 255)
BLUE = (0, 100, 255)
RED = (255, 50, 50)
BLACK = (0, 0, 0)
scale = 120

K = 1 / ( 4 * math.pi * epsilon_0)

charges= [ ( [1,-1], 1 ),( [-1,1], -5 ),  ]  #ใส่ พิกัดเริ่มต้นของประจุ เเละ ขนาดประจุ 
dragged_charge = None
def to_screen(x, y):
    return int(Width // 2 + x * scale), int(Height // 2 - y * scale)

def to_world(x, y):
    return (x - Width // 2) / scale, (Height // 2 - y) / scale

def Electric_Field(px,py):
    Ex,Ey = 0, 0 
    for (chargeX,chargeY), q in charges:
        dx, dy = px - chargeX, py - chargeY
        R_squared = dx**2 + dy**2 
        if R_squared == 0: # Ts would be crack
            continue  
        R = np.sqrt(R_squared)
        Ex += ( K * q  * dx ) / R**3
        Ey += ( K * q  * dy ) / R**3
    return Ex,Ey

def compute_dynamic_threshold(charges, alpha=0.0012, reference_distance=0.5):
    max_q = max(abs(q) for (_, q) in charges)
    threshold = alpha * K * max_q / (reference_distance**2)
    return threshold


def find_equilibrium_points(threshold=1e9, spacing=0.05):
    
    x_vals = np.arange(-Width/(2*scale), Width/(2*scale), spacing)
    y_vals = np.arange(-Height/(2*scale), Height/(2*scale), spacing)
    X, Y = np.meshgrid(x_vals, y_vals)
    
   
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
  
    Ex_total = np.zeros_like(X_flat)
    Ey_total = np.zeros_like(Y_flat)
    
    for (chargeX, chargeY), q in charges:
        dx = X_flat - chargeX
        dy = Y_flat - chargeY
        R_squared = dx**2 + dy**2
        
        
        mask = R_squared != 0
        R = np.zeros_like(R_squared)
        R[mask] = np.sqrt(R_squared[mask])
        
        Ex_total[mask] += (K * q * dx[mask]) / (R[mask]**3)
        Ey_total[mask] += (K * q * dy[mask]) / (R[mask]**3)
    
    E_magnitude = np.sqrt(Ex_total**2 + Ey_total**2)
    
    
    points = list(zip(X_flat[E_magnitude < threshold], Y_flat[E_magnitude < threshold]))
    return points


def draw_arrow(surface, color, start, end, arrow_size=5):
    pygame.draw.line(surface, color, start, end, 1)

    
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)


    left = (
        end[0] - arrow_size * math.cos(angle - math.pi / 6),
        end[1] - arrow_size * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - arrow_size * math.cos(angle + math.pi / 6),
        end[1] - arrow_size * math.sin(angle + math.pi / 6),
    )

    pygame.draw.line(surface, color, end, left, 1)
    pygame.draw.line(surface, color, end, right, 1)


running = True
clock = pygame.time.Clock()
equilibrium_points = find_equilibrium_points()
charges_changed = True
equilibrium_points = []

while running:
    screen.fill(BLACK)
    if charges_changed:
        threshold = compute_dynamic_threshold(charges)
        equilibrium_points = find_equilibrium_points(threshold=threshold, spacing=0.05)
        charges_changed = False
    
    for x in range(0, Width , 40 ):
        for y in range(0, Height, 40):
            Wx,Wy =to_world(x,y)
            Ex, Ey = Electric_Field(Wx, Wy)
            Vect = np.sqrt(Ex**2 + Ey**2)
            if Vect == 0:
                continue 
            Fx = Ex *30 / Vect 
            Fy = Ey *30 / Vect 
            start = (x, y)
            end = (x + Fx, y - Fy)
            draw_arrow(screen, WHITE, start, end)
            
    for x, y in equilibrium_points:
                px, py = to_screen(x, y)
                square_size = 4 
                rect = pygame.Rect(px - square_size//2, py - square_size//2, square_size, square_size)
                pygame.draw.rect(screen, (0, 255, 255), rect)

    for index,( (x,y), q ) in enumerate(charges):
        if q > 0:
            color = BLUE
        else:
            color = RED
        pos = to_screen(x,y)
        pygame.draw.circle(screen, color, pos, 20) #create a circle like particle 
        if dragged_charge == index:
            pygame.draw.circle(screen, (0, 255, 0), pos, 23, 1) #create a circle outside of charge when moving the particle
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            for idx, ((x, y), _) in enumerate(charges):
                sx, sy = to_screen(x, y)
                if (mx - sx)**2 + (my - sy)**2 < 15**2:
                    dragged_charge = idx
                    

        elif event.type == pygame.MOUSEBUTTONUP:
            dragged_charge = None

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                running = False     

        elif event.type == pygame.MOUSEMOTION:
            if dragged_charge is not None:
                Wx, Wy = to_world(*event.pos)
                charges[dragged_charge][0][0] = Wx
                charges[dragged_charge][0][1] = Wy
                charges_changed = True
                
    pygame.display.flip()
    clock.tick(144)

pygame.quit()


            

        
  





