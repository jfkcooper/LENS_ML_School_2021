import numpy as np
import matplotlib.pyplot as plt
import gym
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

def _height(xs):
    return np.sin(3 * xs) * .45 + .55

def render_mcar(env):
    min_position = env.min_position
    max_position = env.max_position
    max_speed = env.max_speed
    goal_position = env.goal_position
    goal_velocity = env.goal_velocity
    
    carwidth = 40
    carheight = 20
    wheel_r = carheight / 2.5
    
    screen_width = 600
    screen_height = 400
    world_width = max_position - min_position
    scale = screen_width / world_width

    low = np.array([min_position, -max_speed], dtype=np.float32)
    high = np.array([max_position, max_speed], dtype=np.float32)
    
    patches = []
    
    # track
    xs = np.linspace(min_position, max_position, 100)
    ys = _height(xs)
    
    # flag
    flagx = (goal_position - min_position) * scale
    flagy1 = _height(goal_position) * scale
    flagy2 = flagy1 + 50
    
    flag = Polygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)], fill=True, 
                   edgecolor='k', facecolor='yellow')
    patches.append(flag)
    
    # car
    pos = env.state[0]
    car_x = (pos - min_position) * scale
    car_y = _height(pos) * scale

    l, r, t, b = car_x - carwidth / 2, car_x + carwidth / 2, car_y + 2*wheel_r + carheight, car_y + 2*wheel_r
    car = Polygon([(l, b), (l, t), (r, t), (r, b)], fill=True, color='k')
    patches.append(car)

    frontwheel = Circle((car_x + carwidth/2 - wheel_r, car_y + wheel_r), wheel_r, fill=True, color='gray')
    patches.append(frontwheel)

    backwheel = Circle((car_x - carwidth/2 + wheel_r, car_y + wheel_r), wheel_r, fill=True, color='gray')
    patches.append(backwheel)
    
    plt.gcf()
    fig = Figure(figsize=(6, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    
    ax = fig.add_subplot(111)    
    ax.plot((xs - min_position)*scale, ys*scale, c='k', lw=4)
    ax.vlines(flagx, flagy1, flagy2, colors='k', lw=2)
    p = PatchCollection(patches)
    ax.add_collection(p)
    ax.set_xlim([0, screen_width])
    ax.set_ylim([0, screen_height])
    
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    
    # Convert to a NumPy array.
    return np.frombuffer(s, np.uint8).reshape((height, width, 4))