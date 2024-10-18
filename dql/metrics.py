import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pygame
import numpy as np
from collections import deque
import threading
from queue import Queue

class MetricsVisualizer:
    def __init__(self, window_size=100000):
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
        self.screen = pygame.display.set_mode(self.plot_size)
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
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        pygame.quit()
                        return
                
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
                
                clock.tick(30)  # Limit to 30 FPS
            except Exception as e:
                print(f"Error in update loop: {e}")
                continue
    
    def _update_plots(self):
        """Update all plot lines and render to pygame surface"""
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

    def save(self):
            """Destructor to save plots before closing."""
            try:
                # Save the entire figure as one image
                self.fig.savefig('training_metrics.png', bbox_inches='tight') 
                
                # Save each subplot as a separate image
                for i, ax in enumerate(self.axs.flat):
                    # Create a new figure for each subplot
                    fig, ax_new = plt.subplots(figsize=(8, 6), dpi=100)

                    # Plot all lines from the original axis
                    for line in ax.get_lines():
                        ax_new.plot(*line.get_data(), label=line.get_label())

                    ax_new.set_title(ax.get_title())
                    ax_new.set_xlabel(ax.get_xlabel())
                    ax_new.set_ylabel(ax.get_ylabel())
                    ax_new.legend()
                    fig.savefig(f'subplot_{i}.png', bbox_inches='tight')
                    plt.close(fig)  # Close the figure to free memory
                    
                print("Plots saved successfully.")
            except Exception as e:
                print(f"Error saving plots: {e}")