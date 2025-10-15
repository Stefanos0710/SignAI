"""

here comes the history from videos and other functions

"""

from datetime import datetime
import os
import shutil

class HistoryVideos:
    def __init__(self):
        self.path = os.path.join(os.path.dirname(__file__), "videos", "history")
        self.source_path = os.path.join(os.path.dirname(__file__), "videos", "current_video.mp4")

        # setup the history folder
        self.setup()

    def get_video(self):
        """Get the current video path"""
        return self.source_path

    def create_name(self):
        # Get current date and time
        now = datetime.now()

        formatted = now.strftime("%Y-%m-%d_%H_%M_%S")

        return f"{formatted}.mp4"


    def save_video(self):
        """Save the current video to history"""
        # Check if source video exists
        if not os.path.exists(self.source_path):
            print(f"Warning: No video to save at {self.source_path}")
            return False

        # Check if source video has content (minimum 1KB to ensure it has actual frames)
        file_size = os.path.getsize(self.source_path)
        if file_size < 1024:  # Less than 1KB is likely empty or corrupted
            print(f"Warning: Video file is too small ({file_size} bytes), not saving to history")
            return False

        # create a new name for the video
        new_name = self.create_name()

        # destination path
        destination = os.path.join(self.path, new_name)

        try:
            # copy file in the history folder
            shutil.copy2(self.source_path, destination)
            print(f"Video saved to history: {destination}")
            return True
        except Exception as e:
            print(f"Error saving video to history: {e}")
            return False

    def clear_history(self):
        pass

    def setup(self):
        os.makedirs(self.path, exist_ok=True)
