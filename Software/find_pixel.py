from bowl_stimulate_class import *


def main():
    arena = Stimulation_Pipeline(img_offsetx=1920+1280, img_offsety=1080+720)
    print("Dot Projection Started. Enter 'azimuth elevation' (e.g., 90 60). Type 'q' to quit.")

    while True:
        user_input = input("Enter coordinates (azi ele): ").strip()
        if user_input.lower() == 'q':
            print("Exiting...")
            break

        try:
            azimuth, elevation = map(float, user_input.split())
            arena.project_dot_at(azimuth, elevation)
        except ValueError:
            print("Invalid input. Please enter two numbers or 'q' to quit.")

if __name__ == "__main__":
    main()