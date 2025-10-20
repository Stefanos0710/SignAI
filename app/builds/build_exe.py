"""
SignAI EXE Builder Script
This script creates an executable EXE file from the SignAI App
"""

import subprocess
import os
import sys

def add_defender_exclusion():
    """Adds Windows Defender exclusion (requires Admin rights)"""
    app_path = os.getcwd()
    print(f"Adding Windows Defender exclusion for: {app_path}")

    try:
        # PowerShell command to add exclusion
        cmd = f'powershell -Command "Add-MpPreference -ExclusionPath \'{app_path}\'"'
        subprocess.run(cmd, shell=True, check=True)
        print("✓ Windows Defender exclusion added")
        return True
    except Exception as e:
        print(f"✗ Error adding exclusion: {e}")
        print("  Please run the script as Administrator!")
        return False

def clean_build():
    """Deletes old build files"""
    print("\nDeleting old build files...")
    dirs_to_remove = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            import shutil
            shutil.rmtree(dir_name)
            print(f"  ✓ {dir_name} deleted")

def build_exe():
    """Creates the EXE file with PyInstaller"""
    print("\n" + "="*50)
    print("Creating EXE file...")
    print("="*50 + "\n")

    # PyInstaller command
    cmd = [
        'pyinstaller',
        '--name=SignAI',
        '--windowed',  # No console window
        '--onedir',    # All files in one folder
        '--noconfirm', # No confirmation
        '--clean',

        # Add data
        '--add-data=ui/main_window.ui;ui',
        '--add-data=icons;icons',
        '--add-data=style.qss;.',
        '--add-data=settings/settings.json;settings',

        # Hidden imports
        '--hidden-import=PySide6.QtCore',
        '--hidden-import=PySide6.QtGui',
        '--hidden-import=PySide6.QtWidgets',
        '--hidden-import=PySide6.QtUiTools',
        '--hidden-import=cv2',
        '--hidden-import=mediapipe',
        '--hidden-import=numpy',
        '--hidden-import=requests',

        # Exclude modules (to make EXE smaller)
        '--exclude-module=matplotlib',
        '--exclude-module=seaborn',
        '--exclude-module=pandas',
        '--exclude-module=scipy',

        'app.py'
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n" + "="*50)
        print("✓ EXE successfully created!")
        print("="*50)
        print(f"\nThe application is located at: dist\\SignAI\\SignAI.exe")
        print("\nTo start the app, run: dist\\SignAI\\SignAI.exe")
        return True
    except subprocess.CalledProcessError as e:
        print("\n" + "="*50)
        print("✗ Error creating the EXE!")
        print("="*50)
        print(f"\nError: {e}")
        return False

def main():
    print("="*50)
    print("SignAI - EXE Builder")
    print("="*50)

    # Step 1: Windows Defender exclusion
    print("\nStep 1: Windows Defender Exclusion")
    print("-"*50)
    response = input("Do you want to add Windows Defender exclusion? (y/n): ")

    if response.lower() in ['y', 'yes']:
        if not add_defender_exclusion():
            print("\nALTERNATIVE: Please add an exclusion manually:")
            print(f"1. Open Windows Security")
            print(f"2. Go to Virus & Threat Protection > Manage Settings")
            print(f"3. Under 'Exclusions' add this folder:")
            print(f"   {os.getcwd()}")
            input("\nPress Enter when done...")

    # Step 2: Delete build files
    print("\nStep 2: Cleanup")
    print("-"*50)
    clean_build()

    # Step 3: Create EXE
    print("\nStep 3: Create EXE")
    print("-"*50)
    success = build_exe()

    if success:
        print("\n✓ DONE! The SignAI App was successfully created as an EXE.")
    else:
        print("\n✗ There were problems creating the EXE.")
        print("   Please check the error messages above.")

    input("\nPress Enter to exit...")

if __name__ == '__main__':
    main()
