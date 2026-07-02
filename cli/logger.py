"""
CLI Progress Logging and Status Formatting for CRN Research Framework.
"""

import sys
import time
from typing import Optional

class ProgressLogger:
    """
    Tracks and prints structured, premium training progress.
    """
    def __init__(self, algorithm: str, total_episodes: int, total_steps: Optional[int] = None):
        self.algorithm = algorithm
        self.total_episodes = total_episodes
        self.total_steps = total_steps
        self.start_time = time.time()
        self.last_update = 0.0

    def format_time(self, seconds: float) -> str:
        """Format time duration in hh:mm:ss."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def update(
        self, 
        episode: int, 
        step: int, 
        reward: float, 
        throughput: float, 
        ber: float, 
        outage: float,
        force: bool = False
    ):
        """Prints a formatted one-line status update."""
        current_time = time.time()
        # Limit update frequency to 1 Hz unless forced
        if not force and (current_time - self.last_update < 1.0):
            return
        
        self.last_update = current_time
        elapsed = current_time - self.start_time
        
        # Calculate ETA
        if episode > 0:
            total_est = (elapsed / episode) * self.total_episodes
            eta = max(0.0, total_est - elapsed)
            eta_str = self.format_time(eta)
        else:
            eta_str = "--:--:--"

        elapsed_str = self.format_time(elapsed)

        # Print premium formatted status
        sys.stdout.write(
            f"\r[{self.algorithm}] "
            f"Ep: {episode}/{self.total_episodes} | "
            f"Step: {step} | "
            f"Reward: {reward:7.2f} | "
            f"SU Thr: {throughput:5.3f} bps/Hz | "
            f"BER: {ber:8.6f} | "
            f"Outage: {outage:5.4f} | "
            f"Elapsed: {elapsed_str} | "
            f"ETA: {eta_str}"
        )
        sys.stdout.flush()

    def complete(self):
        """Cleans up the line after completion."""
        elapsed = time.time() - self.start_time
        sys.stdout.write(f"\n[{self.algorithm}] Training Completed in {self.format_time(elapsed)}\n")
        sys.stdout.flush()

def print_header(title: str):
    """Print standard section headers."""
    print("\n" + "=" * 80)
    print(f" {title.upper().center(78)} ")
    print("=" * 80)

def print_footer():
    """Print standard footer divider."""
    print("=" * 80 + "\n")
