import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.signal import spectrogram
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualization
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['figure.dpi'] = 100

class FourierCoefficient:
    """Fourier coefficient for a single dimension"""
    def __init__(self, amplitude: float, frequency: float, phase: float):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
    
    def evaluate(self, t: float, use_sine: bool = True) -> float:
        if use_sine:
            return self.amplitude * np.sin(self.frequency * t + self.phase)
        else:
            return self.amplitude * np.cos(self.frequency * t + self.phase)

class Fourier3D:
    """3D Fourier series representation"""
    def __init__(self, x_coeffs, y_coeffs, z_coeffs, name="Curve"):
        self.x_coeffs = x_coeffs
        self.y_coeffs = y_coeffs
        self.z_coeffs = z_coeffs
        self.order = len(x_coeffs)
        self.name = name
    
    def evaluate(self, t: float) -> np.ndarray:
        x = sum(c.evaluate(t, use_sine=True) for c in self.x_coeffs)
        y = sum(c.evaluate(t, use_sine=False) for c in self.y_coeffs)
        z = sum(c.evaluate(t, use_sine=True) for c in self.z_coeffs)
        return np.array([x, y, z])
    
    def evaluate_batch(self, t_array: np.ndarray) -> np.ndarray:
        return np.array([self.evaluate(t) for t in t_array])

class FourierIntersectionSimulator:
    """Simulator for Fourier space intersections with visualization"""
    
    def __init__(self, num_curves=3, order=5):
        self.num_curves = num_curves
        self.order = order
        self.curves = []
        self.intersections = []
        self.generate_random_curves()
    
    def generate_random_curves(self):
        """Generate random Fourier curves"""
        for i in range(self.num_curves):
            x_coeffs = []
            y_coeffs = []
            z_coeffs = []
            
            for _ in range(self.order):
                x_coeffs.append(FourierCoefficient(
                    amplitude=np.random.uniform(0.5, 2.0),
                    frequency=np.random.uniform(0.5, 3.0),
                    phase=np.random.uniform(0, 2*np.pi)
                ))
                y_coeffs.append(FourierCoefficient(
                    amplitude=np.random.uniform(0.5, 2.0),
                    frequency=np.random.uniform(0.5, 3.0),
                    phase=np.random.uniform(0, 2*np.pi)
                ))
                z_coeffs.append(FourierCoefficient(
                    amplitude=np.random.uniform(0.5, 2.0),
                    frequency=np.random.uniform(0.5, 3.0),
                    phase=np.random.uniform(0, 2*np.pi)
                ))
            
            self.curves.append(Fourier3D(x_coeffs, y_coeffs, z_coeffs, f"Curve {i}"))
    
    def find_intersections(self, tolerance=0.05):
        """Find intersections between all curve pairs"""
        self.intersections = []
        
        for i in range(self.num_curves):
            for j in range(i+1, self.num_curves):
                t1_vals = np.linspace(0, 2*np.pi, 200)
                t2_vals = np.linspace(0, 2*np.pi, 200)
                
                for t1 in t1_vals:
                    for t2 in t2_vals:
                        p1 = self.curves[i].evaluate(t1)
                        p2 = self.curves[j].evaluate(t2)
                        dist = np.linalg.norm(p1 - p2)
                        
                        if dist < tolerance:
                            # Check if intersection is unique
                            is_unique = True
                            for existing in self.intersections:
                                if np.linalg.norm(np.array([existing[0], existing[1]]) - np.array([t1, t2])) < 0.1:
                                    is_unique = False
                                    break
                            if is_unique:
                                self.intersections.append({
                                    'curves': (i, j),
                                    't_values': (t1, t2),
                                    'point': (p1 + p2) / 2,
                                    'distance': dist
                                })
        
        return self.intersections

class FourierVisualizer:
    """Handles all visualizations for Fourier space"""
    
    def __init__(self, simulator):
        self.simulator = simulator
        self.colors = plt.cm.viridis(np.linspace(0, 1, simulator.num_curves))
    
    def plot_3d_curves_with_intersections(self):
        """3D plot of curves with intersection points"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot curves
        t = np.linspace(0, 2*np.pi, 500)
        for idx, curve in enumerate(self.simulator.curves):
            points = curve.evaluate_batch(t)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 
                   color=self.colors[idx], linewidth=2, alpha=0.7, label=curve.name)
        
        # Plot intersections
        if self.simulator.intersections:
            intersection_points = np.array([inter['point'] for inter in self.simulator.intersections])
            ax.scatter(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2],
                      c='red', s=100, marker='*', edgecolors='white', linewidth=1.5,
                      label=f'Intersections ({len(intersection_points)})', alpha=0.9)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title('3D Fourier Space Curves with Intersection Points', fontsize=14, pad=20)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_spectrum_dots_2d(self):
        """2D spectrum dots visualization - frequency vs amplitude"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Frequency spectrum for X dimension
        ax = axes[0, 0]
        for idx, curve in enumerate(self.simulator.curves):
            freqs = [c.frequency for c in curve.x_coeffs]
            amps = [c.amplitude for c in curve.x_coeffs]
            phases = [c.phase for c in curve.x_coeffs]
            
            # Create spectrum dots with size based on amplitude and color based on phase
            sizes = np.array(amps) * 200
            colors = phases / (2*np.pi)
            scatter = ax.scatter(freqs, amps, s=sizes, c=colors, 
                               cmap='plasma', alpha=0.7, label=curve.name, edgecolors='white', linewidth=0.5)
            ax.scatter(freqs, amps, s=sizes*1.2, c=colors, cmap='plasma', alpha=0.3)
        
        ax.set_xlabel('Frequency (rad/s)', fontsize=11)
        ax.set_ylabel('Amplitude', fontsize=11)
        ax.set_title('X-Dimension Frequency Spectrum\n(Dot size ∝ Amplitude, Color ∝ Phase)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: 3D spectrum bubble chart
        ax = axes[0, 1]
        for idx, curve in enumerate(self.simulator.curves):
            # Combine all dimensions for a comprehensive view
            all_freqs = []
            all_amps = []
            all_phases = []
            
            for coeff in curve.x_coeffs + curve.y_coeffs + curve.z_coeffs:
                all_freqs.append(coeff.frequency)
                all_amps.append(coeff.amplitude)
                all_phases.append(coeff.phase)
            
            sizes = np.array(all_amps) * 150
            colors = all_phases / (2*np.pi)
            scatter = ax.scatter(all_freqs, all_amps, s=sizes, c=colors, 
                               cmap='coolwarm', alpha=0.6, label=curve.name)
        
        ax.set_xlabel('Frequency (rad/s)', fontsize=11)
        ax.set_ylabel('Amplitude', fontsize=11)
        ax.set_title('3D Spectrum (All Dimensions Combined)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Polar spectrum (phase-amplitude relationship)
        ax = axes[1, 0]
        for idx, curve in enumerate(self.simulator.curves):
            phases = [c.phase for c in curve.x_coeffs + curve.y_coeffs + curve.z_coeffs]
            amps = [c.amplitude for c in curve.x_coeffs + curve.y_coeffs + curve.z_coeffs]
            
            ax.scatter(phases, amps, s=np.array(amps)*150, c=amps, 
                      cmap='plasma', alpha=0.7, label=curve.name, edgecolors='white')
        
        ax.set_xlabel('Phase (radians)', fontsize=11)
        ax.set_ylabel('Amplitude', fontsize=11)
        ax.set_title('Phase-Amplitude Relationship', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Harmonic contribution pie chart
        ax = axes[1, 1]
        harmonic_contributions = []
        labels = []
        colors_list = []
        
        for idx, curve in enumerate(self.simulator.curves):
            total_energy = sum([c.amplitude**2 for c in curve.x_coeffs + curve.y_coeffs + curve.z_coeffs])
            harmonic_contributions.append(total_energy)
            labels.append(curve.name)
            colors_list.append(self.colors[idx])
        
        wedges, texts, autotexts = ax.pie(harmonic_contributions, labels=labels, colors=colors_list,
                                          autopct='%1.1f%%', startangle=90, shadow=True)
        ax.set_title('Harmonic Energy Distribution', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_intersection_density_heatmap(self, grid_size=50):
        """2D heatmap of intersection density"""
        if not self.simulator.intersections:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract intersection points
        points = np.array([inter['point'] for inter in self.simulator.intersections])
        
        # 2D projections
        projections = [
            (0, 1, 'X-Y Plane', axes[0]),
            (0, 2, 'X-Z Plane', axes[1])
        ]
        
        for proj_idx, (dim1, dim2, title, ax) in enumerate(projections):
            # Create 2D histogram
            hist, xedges, yedges = np.histogram2d(points[:, dim1], points[:, dim2], bins=grid_size)
            
            # Plot heatmap
            im = ax.imshow(hist.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                          cmap='hot', aspect='auto', alpha=0.8)
            
            # Overlay points
            ax.scatter(points[:, dim1], points[:, dim2], c='cyan', s=10, alpha=0.5, label='Intersections')
            
            ax.set_xlabel(['X', 'X'][proj_idx], fontsize=11)
            ax.set_ylabel(['Y', 'Z'][proj_idx], fontsize=11)
            ax.set_title(f'Intersection Density - {title}', fontsize=12)
            ax.legend()
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Density')
        
        plt.tight_layout()
        return fig
    
    def plot_time_evolution(self, num_frames=100):
        """Animated time evolution of intersections"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        t_vals = np.linspace(0, 2*np.pi, 500)
        
        # Initialize plots
        lines = []
        for idx, curve in enumerate(self.simulator.curves):
            line, = ax.plot([], [], [], color=self.colors[idx], linewidth=2, alpha=0.7, label=curve.name)
            lines.append(line)
        
        intersection_scatter = ax.scatter([], [], [], c='red', s=80, marker='*', 
                                         edgecolors='white', label='Intersections')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Time Evolution of Fourier Curves')
        ax.legend()
        
        def update(frame):
            t_current = t_vals[frame]
            
            # Update each curve
            for idx, curve in enumerate(self.simulator.curves):
                points = curve.evaluate_batch(t_vals[:frame+1])
                lines[idx].set_data(points[:, 0], points[:, 1])
                lines[idx].set_3d_properties(points[:, 2])
            
            # Find intersections at current time
            current_intersections = []
            for i in range(self.simulator.num_curves):
                for j in range(i+1, self.simulator.num_curves):
                    point_i = self.simulator.curves[i].evaluate(t_current)
                    point_j = self.simulator.curves[j].evaluate(t_current)
                    if np.linalg.norm(point_i - point_j) < 0.1:
                        current_intersections.append((point_i + point_j) / 2)
            
            if current_intersections:
                intersections = np.array(current_intersections)
                intersection_scatter._offsets3d = (intersections[:, 0], intersections[:, 1], intersections[:, 2])
            
            return lines + [intersection_scatter]
        
        anim = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
        return anim
    
    def plot_spectrum_waterfall(self):
        """3D waterfall plot of frequency spectrum over curves"""
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare data
        all_freqs = []
        all_amps = []
        all_curve_indices = []
        
        for idx, curve in enumerate(self.simulator.curves):
            freqs = [c.frequency for c in curve.x_coeffs + curve.y_coeffs + curve.z_coeffs]
            amps = [c.amplitude for c in curve.x_coeffs + curve.y_coeffs + curve.z_coeffs]
            all_freqs.extend(freqs)
            all_amps.extend(amps)
            all_curve_indices.extend([idx] * len(freqs))
        
        # Create waterfall plot
        for idx in range(self.simulator.num_curves):
            mask = np.array(all_curve_indices) == idx
            freqs = np.array(all_freqs)[mask]
            amps = np.array(all_amps)[mask]
            
            # Sort by frequency for better visualization
            sort_idx = np.argsort(freqs)
            freqs = freqs[sort_idx]
            amps = amps[sort_idx]
            
            # Create line at each curve level
            y_pos = np.ones_like(freqs) * idx
            ax.plot(freqs, y_pos, amps, color=self.colors[idx], linewidth=2, marker='o', markersize=4)
            
            # Add connecting lines to base
            for f, y, a in zip(freqs, y_pos, amps):
                ax.plot([f, f], [y, y], [0, a], color=self.colors[idx], alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Frequency (rad/s)', fontsize=11)
        ax.set_ylabel('Curve Index', fontsize=11)
        ax.set_zlabel('Amplitude', fontsize=11)
        ax.set_title('Spectrum Waterfall Plot', fontsize=12)
        ax.view_init(elev=25, azim=-45)
        
        return fig
    
    def plot_energy_distribution(self):
        """Energy distribution across harmonics and curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Bar chart of energy per harmonic
        ax = axes[0]
        x = np.arange(self.simulator.order)
        width = 0.25
        multiplier = 0
        
        for idx, curve in enumerate(self.simulator.curves):
            energies = [c.amplitude**2 for c in curve.x_coeffs]
            offset = width * multiplier
            rects = ax.bar(x + offset, energies, width, label=curve.name, color=self.colors[idx])
            multiplier += 1
        
        ax.set_xlabel('Harmonic Index', fontsize=11)
        ax.set_ylabel('Energy (Amplitude²)', fontsize=11)
        ax.set_title('Energy Distribution per Harmonic', fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'H{i+1}' for i in range(self.simulator.order)])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative energy
        ax = axes[1]
        for idx, curve in enumerate(self.simulator.curves):
            energies = [c.amplitude**2 for c in curve.x_coeffs]
            cumulative = np.cumsum(energies)
            cumulative_percent = 100 * cumulative / cumulative[-1]
            ax.plot(range(1, len(cumulative_percent) + 1), cumulative_percent, 
                   'o-', color=self.colors[idx], linewidth=2, markersize=6, label=curve.name)
        
        ax.set_xlabel('Number of Harmonics', fontsize=11)
        ax.set_ylabel('Cumulative Energy (%)', fontsize=11)
        ax.set_title('Cumulative Energy Convergence', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_spectrogram(self):
        """Spectrogram-like visualization of frequency content"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Generate time series for analysis
        t = np.linspace(0, 2*np.pi, 1000)
        
        for idx, curve in enumerate(self.simulator.curves):
            # Create time series for each dimension
            x_series = np.array([curve.evaluate(ti)[0] for ti in t])
            y_series = np.array([curve.evaluate(ti)[1] for ti in t])
            z_series = np.array([curve.evaluate(ti)[2] for ti in t])
            
            # Compute spectrogram for X dimension
            ax = axes[0]
            f, t_spec, Sxx = spectrogram(x_series, fs=100, nperseg=256)
            im = ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), 
                              cmap='viridis', shading='gouraud', alpha=0.7)
            
            # Plot 2: Combined frequency signature
            ax2 = axes[1]
            freqs_x = [c.frequency for c in curve.x_coeffs]
            amps_x = [c.amplitude for c in curve.x_coeffs]
            freqs_y = [c.frequency for c in curve.y_coeffs]
            amps_y = [c.amplitude for c in curve.y_coeffs]
            freqs_z = [c.frequency for c in curve.z_coeffs]
            amps_z = [c.amplitude for c in curve.z_coeffs]
            
            # Create jittered positions for better visibility
            x_pos = freqs_x
            y_pos = freqs_y
            z_pos = freqs_z
            
            # Scatter plot with bubble size representing amplitude
            size_scale = 200
            ax2.scatter(x_pos, [1]*len(x_pos), s=np.array(amps_x)*size_scale, 
                       c='red', alpha=0.6, edgecolors='white', label='X-dim')
            ax2.scatter(y_pos, [2]*len(y_pos), s=np.array(amps_y)*size_scale, 
                       c='green', alpha=0.6, edgecolors='white', label='Y-dim')
            ax2.scatter(z_pos, [3]*len(z_pos), s=np.array(amps_z)*size_scale, 
                       c='blue', alpha=0.6, edgecolors='white', label='Z-dim')
            
            ax2.set_yticks([1, 2, 3])
            ax2.set_yticklabels(['X-Dim', 'Y-Dim', 'Z-Dim'])
            ax2.set_xlabel('Frequency (rad/s)')
            ax2.set_title('Dimension-wise Frequency Signature')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Frequency (Hz)')
        axes[0].set_title('Spectrogram - X Dimension')
        plt.colorbar(im, ax=axes[0], label='Power (dB)')
        
        plt.tight_layout()
        return fig

# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("FOURIER SPACE INTERSECTION SIMULATOR WITH SPECTRUM VISUALIZATION")
    print("=" * 70)
    
    # Initialize simulator
    simulator = FourierIntersectionSimulator(num_curves=3, order=6)
    visualizer = FourierVisualizer(simulator)
    
    # Find intersections
    print("\n🔍 Finding intersections between curves...")
    intersections = simulator.find_intersections(tolerance=0.05)
    print(f"✅ Found {len(intersections)} intersection points")
    
    # Generate all visualizations
    print("\n📊 Generating visualizations...")
    
    # 1. 3D curves with intersections
    fig1 = visualizer.plot_3d_curves_with_intersections()
    plt.show()
    
    # 2. Spectrum dots visualization
    fig2 = visualizer.plot_spectrum_dots_2d()
    plt.show()
    
    # 3. Intersection density heatmap
    fig3 = visualizer.plot_intersection_density_heatmap(grid_size=40)
    if fig3:
        plt.show()
    
    # 4. Spectrum waterfall plot
    fig4 = visualizer.plot_spectrum_waterfall()
    plt.show()
    
    # 5. Energy distribution
    fig5 = visualizer.plot_energy_distribution()
    plt.show()
    
    # 6. Spectrogram visualization
    fig6 = visualizer.plot_spectrogram()
    plt.show()
    
    # Print analysis summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    print(f"\n📈 Intersection Statistics:")
    print(f"   • Total intersections: {len(intersections)}")
    if intersections:
        distances = [inter['distance'] for inter in intersections]
        print(f"   • Average intersection distance: {np.mean(distances):.4f}")
        print(f"   • Min intersection distance: {np.min(distances):.4f}")
        print(f"   • Max intersection distance: {np.max(distances):.4f}")
    
    print(f"\n🎵 Spectral Analysis:")
    for idx, curve in enumerate(simulator.curves):
        all_amps = [c.amplitude for c in curve.x_coeffs + curve.y_coeffs + curve.z_coeffs]
        all_freqs = [c.frequency for c in curve.x_coeffs + curve.y_coeffs + curve.z_coeffs]
        print(f"\n   Curve {idx}:")
        print(f"      • Average amplitude: {np.mean(all_amps):.3f}")
        print(f"      • Average frequency: {np.mean(all_freqs):.3f}")
        print(f"      • Frequency range: [{np.min(all_freqs):.2f}, {np.max(all_freqs):.2f}]")
        print(f"      • Total energy: {sum([a**2 for a in all_amps]):.3f}")
    
    print("\n" + "=" * 70)
    print("✅ All visualizations generated successfully!")
    print("=" * 70)