import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
from PIL import Image
import time

# Title and introduction
st.title('Celestial Body Simulator')
st.markdown("""
This application simulates the movement and positions of celestial bodies based on Newton's laws of motion and the law of universal gravitation.
You can visualize various celestial phenomena from the solar system to binary star systems and three-body problems.
""")

# Create sidebar
st.sidebar.header('Simulation Settings')

# Color name to HEX format conversion dictionary
COLOR_MAP = {
    'red': '#FF0000',
    'green': '#00FF00',
    'blue': '#0000FF',
    'yellow': '#FFFF00',
    'orange': '#FFA500',
    'purple': '#800080',
    'brown': '#A52A2A',
    'gray': '#808080',
    'black': '#000000'
}

# Celestial body class definition
class CelestialBody:
    """Class representing a celestial object"""
    
    def __init__(self, name, mass, radius, position, velocity, color):
        """
        Parameters:
        name (str): Name of the celestial body
        mass (float): Mass of the body (kg)
        radius (float): Radius of the body (m)
        position (array): Initial position [x, y, z] (m)
        velocity (array): Initial velocity [vx, vy, vz] (m/s)
        color (str): Color of the body
        """
        self.name = name
        self.mass = float(mass)  # Ensure mass is float
        self.radius = float(radius)  # Ensure radius is float
        self.position = np.array(position, dtype=np.float64)  # Ensure position is float64
        self.velocity = np.array(velocity, dtype=np.float64)  # Ensure velocity is float64
        self.acceleration = np.zeros(3, dtype=np.float64)  # Ensure acceleration is float64
        # Convert color name to HEX format
        self.color = COLOR_MAP.get(color, color) if isinstance(color, str) else color
        self.trajectory = [np.copy(self.position)]
        self.initial_position = np.copy(position)
        self.initial_velocity = np.copy(velocity)
        
    def update_trajectory(self):
        """Add current position to trajectory history"""
        self.trajectory.append(np.copy(self.position))
        
    def reset(self):
        """Reset body position and velocity to initial state"""
        self.position = np.copy(self.initial_position)
        self.velocity = np.copy(self.initial_velocity)
        self.acceleration = np.zeros(3, dtype=np.float64)
        self.trajectory = [np.copy(self.position)]
        
    def __str__(self):
        return f"{self.name}: mass={self.mass}kg, position={self.position}m, velocity={self.velocity}m/s"

# Physics engine definition
class PhysicsEngine:
    """Engine for physical calculations of celestial bodies"""
    
    def __init__(self, dt=86400, G=6.67430e-11):
        """
        Parameters:
        dt (float): Time step (seconds)
        G (float): Gravitational constant (m^3 kg^-1 s^-2)
        """
        self.dt = float(dt)  # Ensure dt is float
        self.G = float(G)  # Ensure G is float
        self.initial_dt = float(dt)
        self.initial_G = float(G)
        
    def calculate_acceleration(self, bodies):
        """
        Calculate acceleration due to gravity between all bodies
        
        Parameters:
        bodies (list): List of CelestialBody objects
        """
        # Reset acceleration for all bodies
        for body in bodies:
            body.acceleration = np.zeros(3, dtype=np.float64)
        
        # Calculate gravitational interaction for all body pairs
        for i, body1 in enumerate(bodies):
            for body2 in bodies[i+1:]:
                # Calculate vector and distance between two bodies
                r_vec = body2.position - body1.position
                r = np.linalg.norm(r_vec)
                
                # Skip if distance is zero (collision, etc.)
                if r == 0:
                    continue
                
                # Calculate acceleration based on the law of universal gravitation
                # F = G * m1 * m2 / r^2
                # a1 = F / m1 = G * m2 / r^2
                # a2 = F / m2 = G * m1 / r^2
                force_mag = self.G * body1.mass * body2.mass / (r * r)
                
                # Calculate unit vector
                r_hat = r_vec / r
                
                # Update acceleration for each body (action-reaction principle)
                body1.acceleration += (force_mag / body1.mass) * r_hat
                body2.acceleration -= (force_mag / body2.mass) * r_hat
    
    def update_leapfrog(self, bodies):
        """
        Update position and velocity using the leapfrog method
        
        Parameters:
        bodies (list): List of CelestialBody objects
        """
        # Calculate initial acceleration
        self.calculate_acceleration(bodies)
        
        # Update velocity for all bodies by half step
        for body in bodies:
            # Ensure all calculations are done with float64
            half_step = 0.5 * body.acceleration * self.dt
            body.velocity = body.velocity + half_step  # Explicit addition to avoid type issues
        
        # Update position for all bodies by full step
        for body in bodies:
            # Ensure all calculations are done with float64
            position_step = body.velocity * self.dt
            body.position = body.position + position_step  # Explicit addition to avoid type issues
            body.update_trajectory()
        
        # Calculate new acceleration
        self.calculate_acceleration(bodies)
        
        # Update velocity for all bodies by remaining half step
        for body in bodies:
            # Ensure all calculations are done with float64
            half_step = 0.5 * body.acceleration * self.dt
            body.velocity = body.velocity + half_step  # Explicit addition to avoid type issues
            
    def reset(self):
        """Reset physics engine parameters to initial state"""
        self.dt = self.initial_dt
        self.G = self.initial_G

# Celestial system class definition
class CelestialSystem:
    """Class to manage a system of celestial bodies"""
    
    def __init__(self):
        """Initialize celestial system"""
        self.bodies = []
        
    def add_body(self, body):
        """
        Add a body to the system
        
        Parameters:
        body (CelestialBody): Body to add
        """
        self.bodies.append(body)
        
    def remove_body(self, body_name):
        """
        Remove a body from the system
        
        Parameters:
        body_name (str): Name of the body to remove
        """
        self.bodies = [body for body in self.bodies if body.name != body_name]
        
    def get_body(self, body_name):
        """
        Get a body by name
        
        Parameters:
        body_name (str): Name of the body to get
        
        Returns:
        CelestialBody: Found body, or None if not found
        """
        for body in self.bodies:
            if body.name == body_name:
                return body
        return None
    
    def reset(self):
        """Reset all bodies to initial state"""
        for body in self.bodies:
            body.reset()

# Simulation controller class definition
class SimulationController:
    """Class to control the simulation"""
    
    def __init__(self, physics_engine, celestial_system):
        """
        Parameters:
        physics_engine (PhysicsEngine): Physics calculation engine
        celestial_system (CelestialSystem): Celestial system
        """
        self.physics_engine = physics_engine
        self.celestial_system = celestial_system
        self.is_running = False
        self.time_scale = 1.0
        self.current_time = 0.0
        self.max_time = float('inf')
        
    def start(self):
        """Start simulation"""
        self.is_running = True
        
    def pause(self):
        """Pause simulation"""
        self.is_running = False
        
    def reset(self):
        """Reset simulation"""
        self.current_time = 0.0
        self.celestial_system.reset()
        self.physics_engine.reset()
        
    def set_time_scale(self, scale):
        """
        Adjust time scale
        
        Parameters:
        scale (float): New time scale
        """
        self.time_scale = float(scale)
        
    def set_max_time(self, max_time):
        """
        Set maximum simulation time
        
        Parameters:
        max_time (float): Maximum simulation time (seconds)
        """
        self.max_time = float(max_time)
        
    def step(self):
        """Advance one step"""
        if not self.is_running or self.current_time >= self.max_time:
            return False
        
        # Update position and velocity of bodies using physics engine
        self.physics_engine.update_leapfrog(self.celestial_system.bodies)
        
        # Update current time
        self.current_time += self.physics_engine.dt * self.time_scale
        
        return True

# Visualizer class definition
class Visualizer:
    """Class to visualize simulation results"""
    
    def __init__(self, celestial_system, mode='2D'):
        """
        Parameters:
        celestial_system (CelestialSystem): Celestial system
        mode (str): Display mode ('2D' or '3D')
        """
        self.celestial_system = celestial_system
        self.mode = mode
        self.fig = None
        self.ax = None
        self.show_trajectory = True
        self.trajectory_length = 100  # Length of trajectory to display
        self.show_labels = True
        self.size_scale = 1.0  # Body size scale
        self.auto_scale = True  # Automatic axis scaling
        self.view_limits = None  # Display range
        
    def initialize(self, figsize=(10, 8)):
        """
        Initialize drawing environment
        
        Parameters:
        figsize (tuple): Figure size (width, height)
        """
        self.fig = plt.figure(figsize=figsize)
        
        if self.mode == '2D':
            self.ax = self.fig.add_subplot(111)
        else:  # '3D'
            self.ax = self.fig.add_subplot(111, projection='3d')
            
        # Set title and axis labels
        self.ax.set_title('Celestial Simulation')
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        if self.mode == '3D':
            self.ax.set_zlabel('Z [m]')
            
    def update(self):
        """Update frame"""
        self.ax.clear()
        
        # Reset title and axis labels
        self.ax.set_title('Celestial Simulation')
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        if self.mode == '3D':
            self.ax.set_zlabel('Z [m]')
        
        # Draw all bodies
        for body in self.celestial_system.bodies:
            # Draw body position
            size = max(20, body.radius/1e7) * self.size_scale
            if self.mode == '2D':
                self.ax.scatter(body.position[0], body.position[1], 
                               s=size, color=body.color, label=body.name if self.show_labels else None)
            else:  # '3D'
                self.ax.scatter(body.position[0], body.position[1], body.position[2], 
                               s=size, color=body.color, label=body.name if self.show_labels else None)
            
            # Draw trajectory
            if self.show_trajectory and len(body.trajectory) > 1:
                trajectory = np.array(body.trajectory[-self.trajectory_length:])
                if self.mode == '2D':
                    self.ax.plot(trajectory[:, 0], trajectory[:, 1], color=body.color, alpha=0.5)
                else:  # '3D'
                    self.ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                                color=body.color, alpha=0.5)
        
        # Show legend (only if labels are enabled)
        if self.show_labels:
            self.ax.legend()
        
        # Set axis range
        if self.auto_scale:
            self.ax.autoscale(enable=True, axis='both', tight=True)
        elif self.view_limits is not None:
            self.ax.set_xlim(self.view_limits[0])
            self.ax.set_ylim(self.view_limits[1])
            if self.mode == '3D' and len(self.view_limits) > 2:
                self.ax.set_zlim(self.view_limits[2])
        
    def toggle_mode(self):
        """Toggle 2D/3D display"""
        if self.mode == '2D':
            self.mode = '3D'
        else:
            self.mode = '2D'
        
        # Reinitialize figure
        plt.close(self.fig)
        self.initialize()
        
    def toggle_trajectory(self, show=None):
        """
        Toggle trajectory display
        
        Parameters:
        show (bool): Whether to show trajectory. If None, toggle current state
        """
        if show is None:
            self.show_trajectory = not self.show_trajectory
        else:
            self.show_trajectory = show
            
    def toggle_labels(self, show=None):
        """
        Toggle label display
        
        Parameters:
        show (bool): Whether to show labels. If None, toggle current state
        """
        if show is None:
            self.show_labels = not self.show_labels
        else:
            self.show_labels = show
            
    def set_trajectory_length(self, length):
        """
        Set length of trajectory to display
        
        Parameters:
        length (int): Length of trajectory to display
        """
        self.trajectory_length = int(length)
        
    def set_size_scale(self, scale):
        """
        Set body size scale
        
        Parameters:
        scale (float): Size scale
        """
        self.size_scale = float(scale)
        
    def set_auto_scale(self, auto_scale):
        """
        Set automatic axis scaling
        
        Parameters:
        auto_scale (bool): Whether to enable automatic scaling
        """
        self.auto_scale = auto_scale
        
    def set_view_limits(self, xlim, ylim, zlim=None):
        """
        Set display range
        
        Parameters:
        xlim (tuple): X-axis display range (min, max)
        ylim (tuple): Y-axis display range (min, max)
        zlim (tuple): Z-axis display range (min, max) (3D mode only)
        """
        if zlim is not None:
            self.view_limits = (xlim, ylim, zlim)
        else:
            self.view_limits = (xlim, ylim)

# Functions to create preset scenarios
def create_solar_system():
    """
    Create solar system simulation
    
    Returns:
    CelestialSystem: Solar system
    """
    system = CelestialSystem()
    
    # Sun
    sun = CelestialBody(
        name="Sun",
        mass=1.989e30,  # kg
        radius=6.957e8,  # m
        position=[0, 0, 0],  # m
        velocity=[0, 0, 0],  # m/s
        color="#FFFF00"  # Yellow
    )
    system.add_body(sun)
    
    # Mercury
    mercury = CelestialBody(
        name="Mercury",
        mass=3.301e23,  # kg
        radius=2.44e6,  # m
        position=[5.791e10, 0, 0],  # m
        velocity=[0, 4.74e4, 0],  # m/s
        color="#808080"  # Gray
    )
    system.add_body(mercury)
    
    # Venus
    venus = CelestialBody(
        name="Venus",
        mass=4.867e24,  # kg
        radius=6.052e6,  # m
        position=[1.082e11, 0, 0],  # m
        velocity=[0, 3.5e4, 0],  # m/s
        color="#FFA500"  # Orange
    )
    system.add_body(venus)
    
    # Earth
    earth = CelestialBody(
        name="Earth",
        mass=5.972e24,  # kg
        radius=6.371e6,  # m
        position=[1.496e11, 0, 0],  # m
        velocity=[0, 2.98e4, 0],  # m/s
        color="#0000FF"  # Blue
    )
    system.add_body(earth)
    
    # Mars
    mars = CelestialBody(
        name="Mars",
        mass=6.417e23,  # kg
        radius=3.39e6,  # m
        position=[2.279e11, 0, 0],  # m
        velocity=[0, 2.41e4, 0],  # m/s
        color="#FF0000"  # Red
    )
    system.add_body(mars)
    
    # Jupiter
    jupiter = CelestialBody(
        name="Jupiter",
        mass=1.898e27,  # kg
        radius=6.991e7,  # m
        position=[7.786e11, 0, 0],  # m
        velocity=[0, 1.31e4, 0],  # m/s
        color="#A52A2A"  # Brown
    )
    system.add_body(jupiter)
    
    return system


def create_earth_moon_system():
    """
    Create Earth-Moon system simulation
    
    Returns:
    CelestialSystem: Earth-Moon system
    """
    system = CelestialSystem()
    
    # Earth
    earth = CelestialBody(
        name="Earth",
        mass=5.972e24,  # kg
        radius=6.371e6,  # m
        position=[0, 0, 0],  # m
        velocity=[0, 0, 0],  # m/s
        color="#0000FF"  # Blue
    )
    system.add_body(earth)
    
    # Moon
    moon = CelestialBody(
        name="Moon",
        mass=7.342e22,  # kg
        radius=1.737e6,  # m
        position=[3.844e8, 0, 0],  # m
        velocity=[0, 1.022e3, 0],  # m/s
        color="#808080"  # Gray
    )
    system.add_body(moon)
    
    return system


def create_binary_star_system():
    """
    Create binary star system simulation
    
    Returns:
    CelestialSystem: Binary star system
    """
    system = CelestialSystem()
    
    # Star 1
    star1 = CelestialBody(
        name="Star 1",
        mass=1.5e30,  # kg
        radius=7e8,  # m
        position=[3e11, 0, 0],  # m
        velocity=[0, 2e4, 0],  # m/s
        color="#FFFF00"  # Yellow
    )
    system.add_body(star1)
    
    # Star 2
    star2 = CelestialBody(
        name="Star 2",
        mass=1.0e30,  # kg
        radius=5e8,  # m
        position=[-3e11, 0, 0],  # m
        velocity=[0, -3e4, 0],  # m/s
        color="#FFA500"  # Orange
    )
    system.add_body(star2)
    
    return system


def create_three_body_system():
    """
    Create three-body problem simulation
    
    Returns:
    CelestialSystem: Three-body system
    """
    system = CelestialSystem()
    
    # Body 1
    body1 = CelestialBody(
        name="Body 1",
        mass=1.0e30,  # kg
        radius=5e8,  # m
        position=[3e11, 0, 0],  # m
        velocity=[0, 2e4, 0],  # m/s
        color="#FF0000"  # Red
    )
    system.add_body(body1)
    
    # Body 2
    body2 = CelestialBody(
        name="Body 2",
        mass=1.0e30,  # kg
        radius=5e8,  # m
        position=[-3e11, 0, 0],  # m
        velocity=[0, -2e4, 0],  # m/s
        color="#0000FF"  # Blue
    )
    system.add_body(body2)
    
    # Body 3
    body3 = CelestialBody(
        name="Body 3",
        mass=5.0e29,  # kg
        radius=3e8,  # m
        position=[0, 4e11, 0],  # m
        velocity=[-1.5e4, 0, 0],  # m/s
        color="#00FF00"  # Green
    )
    system.add_body(body3)
    
    return system

# Function to convert animation to GIF
def create_animation_gif(simulation_controller, visualizer, frames=100, interval=50):
    """
    Create animation GIF
    
    Parameters:
    simulation_controller (SimulationController): Simulation control object
    visualizer (Visualizer): Visualization object
    frames (int): Number of animation frames
    interval (int): Time between frames (milliseconds)
    
    Returns:
    bytes: GIF data
    """
    # Reset simulation to initial state
    simulation_controller.reset()
    simulation_controller.start()
    
    # Create a deep copy of the simulation controller to avoid modifying the original
    import copy
    sim_copy = copy.deepcopy(simulation_controller)
    
    # List to store frames
    frame_images = []
    
    try:
        # Generate each frame
        for _ in range(int(frames)):
            # Advance simulation one step
            sim_copy.step()
            
            # Update visualization
            visualizer.update()
            
            # Convert figure to byte stream
            buf = io.BytesIO()
            visualizer.fig.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert to PIL image
            img = Image.open(buf)
            frame_images.append(img)
        
        # Create GIF
        gif_buf = io.BytesIO()
        frame_images[0].save(
            gif_buf, 
            format='GIF', 
            save_all=True, 
            append_images=frame_images[1:], 
            duration=int(interval), 
            loop=0
        )
        gif_buf.seek(0)
        
        return gif_buf.getvalue()
    except Exception as e:
        st.error(f"Error creating animation: {str(e)}")
        return None

# Main part of Streamlit app
def main():
    # Initialize session state
    if 'celestial_system' not in st.session_state:
        st.session_state.celestial_system = create_solar_system()
    
    if 'physics_engine' not in st.session_state:
        st.session_state.physics_engine = PhysicsEngine(dt=86400)  # 1 day step
    
    if 'simulation_controller' not in st.session_state:
        st.session_state.simulation_controller = SimulationController(
            st.session_state.physics_engine, 
            st.session_state.celestial_system
        )
    
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer(st.session_state.celestial_system, mode='2D')
        st.session_state.visualizer.initialize()
    
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    
    if 'current_scenario' not in st.session_state:
        st.session_state.current_scenario = 'solar_system'
    
    # Scenario selection
    scenario = st.sidebar.radio(
        "Select Scenario",
        ('Solar System', 'Earth-Moon System', 'Binary Star System', 'Three-Body Problem', 'Custom'),
        index=0 if st.session_state.current_scenario == 'solar_system' else
               1 if st.session_state.current_scenario == 'earth_moon' else
               2 if st.session_state.current_scenario == 'binary_star' else
               3 if st.session_state.current_scenario == 'three_body' else 4
    )
    
    # If scenario changed
    scenario_map = {
        'Solar System': 'solar_system',
        'Earth-Moon System': 'earth_moon',
        'Binary Star System': 'binary_star',
        'Three-Body Problem': 'three_body',
        'Custom': 'custom'
    }
    
    if st.session_state.current_scenario != scenario_map[scenario]:
        if scenario == 'Solar System':
            st.session_state.celestial_system = create_solar_system()
        elif scenario == 'Earth-Moon System':
            st.session_state.celestial_system = create_earth_moon_system()
        elif scenario == 'Binary Star System':
            st.session_state.celestial_system = create_binary_star_system()
        elif scenario == 'Three-Body Problem':
            st.session_state.celestial_system = create_three_body_system()
        # Do nothing for Custom
        
        # Update simulation controller and visualizer
        st.session_state.simulation_controller = SimulationController(
            st.session_state.physics_engine, 
            st.session_state.celestial_system
        )
        st.session_state.visualizer.celestial_system = st.session_state.celestial_system
        
        # Update current scenario
        st.session_state.current_scenario = scenario_map[scenario]
    
    # Simulation settings
    st.sidebar.subheader('Simulation Settings')
    
    # Time step
    dt_days = st.sidebar.slider('Time Step (days)', 0.1, 10.0, 1.0)
    st.session_state.physics_engine.dt = float(dt_days * 86400)  # Convert days to seconds
    
    # Time scale
    time_scale = st.sidebar.slider('Time Scale', 0.1, 10.0, 1.0)
    st.session_state.simulation_controller.set_time_scale(time_scale)
    
    # Display settings
    st.sidebar.subheader('Display Settings')
    
    # 2D/3D display
    display_mode = st.sidebar.radio('Display Mode', ('2D', '3D'))
    if st.session_state.visualizer.mode != display_mode:
        st.session_state.visualizer.mode = display_mode
        plt.close(st.session_state.visualizer.fig)
        st.session_state.visualizer.initialize()
    
    # Show trajectory
    show_trajectory = st.sidebar.checkbox('Show Trajectory', value=True)
    st.session_state.visualizer.toggle_trajectory(show_trajectory)
    
    # Show labels
    show_labels = st.sidebar.checkbox('Show Labels', value=True)
    st.session_state.visualizer.toggle_labels(show_labels)
    
    # Trajectory length
    trajectory_length = st.sidebar.slider('Trajectory Length', 10, 1000, 100)
    st.session_state.visualizer.set_trajectory_length(trajectory_length)
    
    # Body size
    size_scale = st.sidebar.slider('Body Size', 0.1, 10.0, 1.0)
    st.session_state.visualizer.set_size_scale(size_scale)
    
    # Simulation control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button('Start', key='start_button'):
            st.session_state.is_running = True
            st.session_state.simulation_controller.start()
    
    with col2:
        if st.button('Pause', key='pause_button'):
            st.session_state.is_running = False
            st.session_state.simulation_controller.pause()
    
    with col3:
        if st.button('Reset', key='reset_button'):
            st.session_state.is_running = False
            st.session_state.simulation_controller.reset()
    
    # Body parameters display and editing
    st.subheader('Body Parameters')
    
    # Body list tabs
    body_tabs = st.tabs([body.name for body in st.session_state.celestial_system.bodies] + ["Add New"])
    
    # Edit existing bodies
    for i, tab in enumerate(body_tabs[:-1]):  # Exclude last tab (Add New)
        with tab:
            body = st.session_state.celestial_system.bodies[i]
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_name = st.text_input('Name', value=body.name, key=f'name_{i}')
                new_mass = st.number_input('Mass (kg)', value=float(body.mass), format='%e', key=f'mass_{i}')
                new_radius = st.number_input('Radius (m)', value=float(body.radius), format='%e', key=f'radius_{i}')
                # Unify color selection to HEX format
                new_color = st.color_picker('Color', value=body.color, key=f'color_{i}')
            
            with col2:
                st.write('Position (m)')
                new_pos_x = st.number_input('X', value=float(body.position[0]), format='%e', key=f'pos_x_{i}')
                new_pos_y = st.number_input('Y', value=float(body.position[1]), format='%e', key=f'pos_y_{i}')
                new_pos_z = st.number_input('Z', value=float(body.position[2]), format='%e', key=f'pos_z_{i}')
                
                st.write('Velocity (m/s)')
                new_vel_x = st.number_input('X', value=float(body.velocity[0]), format='%e', key=f'vel_x_{i}')
                new_vel_y = st.number_input('Y', value=float(body.velocity[1]), format='%e', key=f'vel_y_{i}')
                new_vel_z = st.number_input('Z', value=float(body.velocity[2]), format='%e', key=f'vel_z_{i}')
            
            # Update button
            if st.button('Update', key=f'update_{i}'):
                body.name = new_name
                body.mass = float(new_mass)
                body.radius = float(new_radius)
                body.color = new_color
                body.position = np.array([new_pos_x, new_pos_y, new_pos_z], dtype=np.float64)
                body.velocity = np.array([new_vel_x, new_vel_y, new_vel_z], dtype=np.float64)
                body.initial_position = np.copy(body.position)
                body.initial_velocity = np.copy(body.velocity)
                st.success(f'Body "{new_name}" updated.')
            
            # Delete button
            if st.button('Delete', key=f'delete_{i}'):
                st.session_state.celestial_system.remove_body(body.name)
                st.success(f'Body "{body.name}" deleted.')
                st.experimental_rerun()
    
    # Add new body
    with body_tabs[-1]:
        col1, col2 = st.columns(2)
        
        with col1:
            new_name = st.text_input('Name', value='New Body', key='new_name')
            new_mass = st.number_input('Mass (kg)', value=1.0e24, format='%e', key='new_mass')
            new_radius = st.number_input('Radius (m)', value=6.0e6, format='%e', key='new_radius')
            # Unify color selection to HEX format
            new_color = st.color_picker('Color', value='#00FF00', key='new_color')
        
        with col2:
            st.write('Position (m)')
            new_pos_x = st.number_input('X', value=2.0e11, format='%e', key='new_pos_x')
            new_pos_y = st.number_input('Y', value=0.0, format='%e', key='new_pos_y')
            new_pos_z = st.number_input('Z', value=0.0, format='%e', key='new_pos_z')
            
            st.write('Velocity (m/s)')
            new_vel_x = st.number_input('X', value=0.0, format='%e', key='new_vel_x')
            new_vel_y = st.number_input('Y', value=2.0e4, format='%e', key='new_vel_y')
            new_vel_z = st.number_input('Z', value=0.0, format='%e', key='new_vel_z')
        
        # Add button
        if st.button('Add', key='add_new_body'):
            # Check if body with same name exists
            if any(body.name == new_name for body in st.session_state.celestial_system.bodies):
                st.error(f'A body named "{new_name}" already exists.')
            else:
                new_body = CelestialBody(
                    name=new_name,
                    mass=float(new_mass),
                    radius=float(new_radius),
                    position=[float(new_pos_x), float(new_pos_y), float(new_pos_z)],
                    velocity=[float(new_vel_x), float(new_vel_y), float(new_vel_z)],
                    color=new_color
                )
                st.session_state.celestial_system.add_body(new_body)
                st.success(f'New body "{new_name}" added.')
                st.experimental_rerun()
    
    # Simulation display
    st.subheader('Simulation')
    
    # Create placeholder
    simulation_placeholder = st.empty()
    
    # Run simulation
    if st.session_state.is_running:
        # Execute a fixed number of steps
        for _ in range(10):  # Update display every 10 steps
            st.session_state.simulation_controller.step()
    
    # Update visualization
    st.session_state.visualizer.update()
    
    # Display figure
    simulation_placeholder.pyplot(st.session_state.visualizer.fig)
    
    # Animation GIF generation button
    st.subheader('Generate Animation GIF')
    
    col1, col2 = st.columns(2)
    
    with col1:
        frames = st.number_input('Number of Frames', min_value=10, max_value=500, value=100, key='gif_frames')
    
    with col2:
        interval = st.number_input('Frame Interval (ms)', min_value=10, max_value=500, value=50, key='gif_interval')
    
    if st.button('Generate Animation GIF', key='generate_gif'):
        with st.spinner('Generating animation...'):
            # Save current state
            current_running = st.session_state.is_running
            
            # Generate animation
            gif_data = create_animation_gif(
                st.session_state.simulation_controller,
                st.session_state.visualizer,
                frames=int(frames),
                interval=int(interval)
            )
            
            if gif_data:
                # Base64 encode
                b64 = base64.b64encode(gif_data).decode()
                
                # Display with HTML
                st.markdown(
                    f'<img src="data:image/gif;base64,{b64}" alt="animation">',
                    unsafe_allow_html=True
                )
                
                # Download button
                st.download_button(
                    label="Download GIF",
                    data=gif_data,
                    file_name="celestial_simulation.gif",
                    mime="image/gif",
                    key='download_gif'
                )
            
            # Restore original state
            if current_running:
                st.session_state.is_running = True
                st.session_state.simulation_controller.start()
            else:
                st.session_state.is_running = False
                st.session_state.simulation_controller.pause()
    
    # Footer
    st.markdown('---')
    st.markdown('© 2025 Celestial Body Simulator')

if __name__ == '__main__':
    main()
