import subprocess
import sys
import os

def main():
    print("=" * 40)
    print("  Doomscroll: Skyrim Edition")
    print("=" * 40)
    print()
    print("Select your platform:")
    print("  1) Mac")
    print("  2) Windows")
    print()

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        platform_dir = "mac"
    elif choice == "2":
        platform_dir = "windows"
    else:
        print("Invalid choice. Please run again and enter 1 or 2.")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements = os.path.join(script_dir, platform_dir, "requirements.txt")
    main_script = os.path.join(script_dir, platform_dir, "main.py")

    print(f"\nInstalling dependencies from {platform_dir}/requirements.txt ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements])

    print(f"\nLaunching {platform_dir}/main.py ...\n")
    os.chdir(script_dir)
    subprocess.check_call([sys.executable, main_script])


if __name__ == "__main__":
    main()
