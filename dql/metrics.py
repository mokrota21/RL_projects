import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pygame
import numpy as np
from collections import deque
import threading
from queue import Queue
import json
import os

class MetricsVisualizer:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        
        # Metrics storage
        self.losses = deque(maxlen=window_size)
        self.avg_rewards = deque(maxlen=window_size)
        self.goals = deque(maxlen=window_size)
        self.subgoals = deque(maxlen=window_size)
        self.epsilon_values = deque(maxlen=window_size)
        
        # Setup pygame
        pygame.init()
        self.plot_size = (800, 600)
        # Add window flags to handle focus events better
        self.screen = pygame.display.set_mode(self.plot_size, pygame.RESIZABLE | pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('Training Metrics')
        
        # Setup the plot
        plt.style.use('dark_background')
        self.fig, self.axs = plt.subplots(2, 2, figsize=(8, 6), dpi=100)
        self.fig.suptitle('Training Metrics', fontsize=16)
        
        # Initialize lines
        self.lines = {}
        self.setup_subplots()
        
        # Communication queue for thread-safe updates
        self.metrics_queue = Queue()
        self.event_queue = Queue()  # New queue for handling window events
        self.running = True
        
        # For converting matplotlib to pygame surface
        self.canvas = agg.FigureCanvasAgg(self.fig)
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
    def setup_subplots(self):
        # Loss subplot
        self.axs[0, 0].set_title('Training Loss')
        self.axs[0, 0].set_xlabel('Steps')
        self.axs[0, 0].set_ylabel('Loss')
        self.lines['loss'], = self.axs[0, 0].plot([], [], 'r-', label='Loss')
        self.axs[0, 0].grid(True, alpha=0.3)
        
        # Average reward subplot
        self.axs[0, 1].set_title('Average Reward')
        self.axs[0, 1].set_xlabel('Episodes')
        self.axs[0, 1].set_ylabel('Reward')
        self.lines['reward'], = self.axs[0, 1].plot([], [], 'g-', label='Avg Reward')
        self.axs[0, 1].grid(True, alpha=0.3)
        
        # Success rate subplot
        self.axs[1, 0].set_title('Success Rate')
        self.axs[1, 0].set_xlabel('Episodes')
        self.axs[1, 0].set_ylabel('Rate')
        self.lines['goal'], = self.axs[1, 0].plot([], [], 'r-', label='Goal')
        self.lines['subgoal'], = self.axs[1, 0].plot([], [], 'b-', label='Subgoal')
        self.axs[1, 0].grid(True, alpha=0.3)
        
        # Epsilon subplot
        self.axs[1, 1].set_title('Exploration Rate (ε)')
        self.axs[1, 1].set_xlabel('Episodes')
        self.axs[1, 1].set_ylabel('ε')
        self.lines['epsilon'], = self.axs[1, 1].plot([], [], 'y-', label='ε')
        self.axs[1, 1].grid(True, alpha=0.3)
        
        # Add legends
        for ax in self.axs.flat:
            ax.legend()
        
        plt.tight_layout()
    
    def update_metrics(self, loss=None, reward=None, goal=None, subgoal=None, epsilon=None):
        """Thread-safe method to update metrics"""
        self.metrics_queue.put({
            'loss': loss,
            'reward': reward,
            'goal': goal,
            'subgoal': subgoal,
            'epsilon': epsilon
        })
    
    def _update_loop(self):
        """Background thread for updating plots"""
        clock = pygame.time.Clock()
        
        while self.running:
            try:
                # Handle events in the main thread
                if not self._handle_events():
                    break

                # Update metrics
                updated = False
                while not self.metrics_queue.empty():
                    metrics = self.metrics_queue.get()
                    
                    if metrics['loss'] is not None:
                        self.losses.append(float(metrics['loss']))
                        updated = True
                    if metrics['reward'] is not None:
                        self.avg_rewards.append(float(metrics['reward']))
                        updated = True
                    if metrics['goal'] is not None:
                        self.goals.append(float(metrics['goal']))
                        updated = True
                    if metrics['subgoal'] is not None:
                        self.subgoals.append(float(metrics['subgoal']))
                        updated = True
                    if metrics['epsilon'] is not None:
                        self.epsilon_values.append(float(metrics['epsilon']))
                        updated = True
                
                # Update plots if needed
                if updated:
                    self._update_plots()
                
                # Limit frame rate but allow for event processing
                clock.tick(30)
                pygame.time.wait(10)  # Small delay to prevent high CPU usage
                
            except Exception as e:
                print(f"Error in update loop: {e}")
                continue
    
    def _update_plots(self):
        """Update all plot lines and render to pygame surface"""
        if not self.running:
            return

        try:
            # Update loss plot
            if len(self.losses) > 0:
                self.lines['loss'].set_data(range(len(self.losses)), self.losses)
                self.axs[0, 0].relim()
                self.axs[0, 0].autoscale_view()
            
            # Update reward plot
            if len(self.avg_rewards) > 0:
                self.lines['reward'].set_data(range(len(self.avg_rewards)), self.avg_rewards)
                self.axs[0, 1].relim()
                self.axs[0, 1].autoscale_view()
            
            # Update success rate plot
            if len(self.goals) > 0:
                self.lines['goal'].set_data(range(len(self.goals)), self.goals)
                self.axs[1, 0].relim()
                self.axs[1, 0].autoscale_view()
            
            if len(self.subgoals) > 0:   
                self.lines['subgoal'].set_data(range(len(self.subgoals)), self.subgoals)
                self.axs[1, 0].relim()
                self.axs[1, 0].autoscale_view()
            
            # Update epsilon plot
            if len(self.epsilon_values) > 0:
                self.lines['epsilon'].set_data(range(len(self.epsilon_values)), self.epsilon_values)
                self.axs[1, 1].relim()
                self.axs[1, 1].autoscale_view()
            
            # Draw the plot
            self.canvas.draw()
            
            # Get the RGBA buffer from the figure
            buf = self.canvas.buffer_rgba()
            arr = np.asarray(buf)
            
            # Convert to pygame surface
            width, height = self.canvas.get_width_height()
            surf = pygame.image.frombuffer(arr.tobytes(), (width, height), 'RGBA')
            
            # Display the plot
            scaled_surf = pygame.transform.scale(surf, self.plot_size)
            self.screen.blit(scaled_surf, (0, 0))
            pygame.display.flip()
            
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    def close(self):
        """Cleanup method"""
        self.running = False
        pygame.quit()
        plt.close(self.fig)

    def _get_unique_filename(self, base_path):
        """
        Generate a unique filename by adding a numeric suffix if the file already exists.
        
        Args:
            base_path (str): The base file path to check
            
        Returns:
            str: A unique file path
        """
        if not os.path.exists(base_path):
            return base_path
            
        # Split the base_path into name and extension
        name, ext = os.path.splitext(base_path)
        counter = 1
        
        # Keep trying new numbers until we find an unused filename
        while True:
            new_path = f"{name}_{counter}{ext}"
            if not os.path.exists(new_path):
                return new_path
            counter += 1

    def save(self, filename='metrics_data.json'):
        """
        Save all metrics data to a JSON file. If the file already exists,
        a new file with a numeric suffix will be created.
        
        Args:
            filename (str): The name of the file to save the metrics to
            
        Returns:
            str: The actual filename where the data was saved
        """
        try:
            # Get unique filenames for all files we'll save
            json_path = self._get_unique_filename(filename)
            base_name = os.path.splitext(json_path)[0]
            plot_path = self._get_unique_filename(f"{base_name}_full.png")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(json_path) if os.path.dirname(json_path) else '.', exist_ok=True)
            
            # Prepare metrics data
            metrics_data = {
                'losses': list(self.losses),
                'avg_rewards': list(self.avg_rewards),
                'goals': list(self.goals),
                'subgoals': list(self.subgoals),
                'epsilon_values': list(self.epsilon_values),
            }
            
            # Save JSON data
            with open(json_path, 'w') as f:
                json.dump(metrics_data, f, indent=4)
            
            return json_path
            
        except Exception as e:
            print(f"Error saving metrics data: {e}")
            return None

    @classmethod
    def load(cls, filename='metrics_data.json'):
        """
        Load metrics data from a JSON file and create a new visualizer instance.
        
        Args:
            filename (str): The name of the file to load the metrics from
            
        Returns:
            MetricsVisualizer: A new instance with the loaded metrics
        """
        try:
            with open(filename, 'r') as f:
                metrics_data = json.load(f)
            
            # Create new instance with saved window size
            visualizer = cls(window_size=metrics_data['window_size'])
            
            # Load metrics into deques
            visualizer.losses.extend(metrics_data['losses'])
            visualizer.avg_rewards.extend(metrics_data['avg_rewards'])
            visualizer.goals.extend(metrics_data['goals'])
            visualizer.subgoals.extend(metrics_data['subgoals'])
            visualizer.epsilon_values.extend(metrics_data['epsilon_values'])
            
            # Force an update of the plots
            visualizer._update_plots()
            
            print(f"Metrics data loaded successfully from {filename}")
            print(f"Data timestamp: {metrics_data['timestamp']}")
            
            # Print information about associated plot files
            if 'plot_files' in metrics_data:
                print("\nAssociated plot files:")
                print(f"- Full plot: {metrics_data['plot_files']['full_plot']}")
                print("- Subplots:")
                for subplot in metrics_data['plot_files']['subplots']:
                    print(f"  - {subplot}")
            
            return visualizer
            
        except Exception as e:
            print(f"Error loading metrics data: {e}")
            return None
    
    def _handle_events(self):
        """Handle pygame events in the main thread"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                return False
            elif event.type == pygame.VIDEORESIZE:
                self.plot_size = (event.w, event.h)
                self.screen = pygame.display.set_mode(self.plot_size, pygame.RESIZABLE | pygame.HWSURFACE | pygame.DOUBLEBUF)
            elif event.type == pygame.ACTIVEEVENT:
                # Force a redraw when window gets focus
                if event.gain:
                    self._update_plots()
        return True