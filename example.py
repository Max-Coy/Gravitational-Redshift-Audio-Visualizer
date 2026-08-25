from pathlib import Path

from src.redshift_animation import RedshiftAnimation
from src.core.config import RedshiftConfig

ROOT = Path(__file__).parent


def main():
    config = RedshiftConfig()
    config.print_local_progress = True
    config.cores = 1

    animation = RedshiftAnimation(
        input_path = ROOT / "assets" / "example.wav", # Input path 
        output_path = ROOT / "example.mp4",
        config = config,
        )

    # a, n = animation.check_frame_pollution()
    animation.optimize_threshold(thresh_percentage = 0.5)
    # max_f, avg_f = animation.check_frame_pollution()

if __name__ == "__main__":
    main()
