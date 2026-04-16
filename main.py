"""
Entry point for the Thermal Tracker AR system.

Run with:
    python main.py                  — Full pipeline demo
    python main.py diffusion        — Heat diffusion visualisation
    python main.py motion           — Camera motion + tracking demo
"""

import sys


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "diffusion":
        from demos.heat_diffusion_demo import run

        run()
    elif mode == "motion":
        from demos.motion_tracking_demo import run

        run()
    else:
        from demos.full_pipeline_demo import run_demo

        run_demo()


if __name__ == "__main__":
    main()
