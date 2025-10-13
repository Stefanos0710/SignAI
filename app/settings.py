import os
import json

class Settings:
    def __init__(self):
        self.folder = "settings"
        self.settings_url = f"{self.folder}/settings.json"

        # set up the settings file
        self.setup()

        # load existing settings or use defaults
        self.load()

    def save(self):
        settings = {
            "debug": self.debug,
            "history": self.history
        }

        settingsfile = os.path.join(self.folder, "settings.json")

        # save settings
        with open(settingsfile, "w") as f:
            json.dump(settings, f, indent=4)
        print(f"Settings saved: {settings}")

    def load(self):
        settingsfile = os.path.join(self.folder, "settings.json")

        try:
            with open(settingsfile, "r") as f:
                settings = json.load(f)
                self.debug = settings.get("debug", False)
                self.history = settings.get("history", True)
            print(f"Settings loaded: debug={self.debug}, history={self.history}")
        except (FileNotFoundError, json.JSONDecodeError):
            # Use defaults if file doesn't exist or is corrupted
            self.debug = False
            self.history = True
            self.save()  # Save defaults
            print("Settings initialized with defaults")

    def setup(self):
        os.makedirs(self.folder, exist_ok=True)
