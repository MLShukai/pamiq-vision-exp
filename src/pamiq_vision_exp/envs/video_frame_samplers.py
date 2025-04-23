import random
from pathlib import Path

import cv2
import torch


class RandomVideoFrameSampler:
    """Generate random image frames from video files in a specified folder.

    This class loads video files from a folder, selects videos based on
    their frame count probabilities, and returns random frames as
    tensors.
    """

    def __init__(
        self,
        folder: str | Path,
        extensions: list[str] = ["mp4", "avi"],
        max_frames_per_video: int | None = None,
    ) -> None:
        """Initialize the image generator.

        Args:
            folder: Path to the folder containing video files
            extensions: List of video file extensions to consider
            max_frames_per_video: Maximum number of frames to read from a single video

        Raises:
            ValueError: If no video files are found or if total frame count is 0
        """
        self.folder = Path(folder)
        self.extensions = extensions
        self.max_frames_per_video = max_frames_per_video

        # Find all video files in the folder
        video_files = []
        for ext in extensions:
            video_files.extend(list(self.folder.glob(f"*.{ext}")))

        if len(video_files) == 0:
            raise ValueError(
                f"No video files with extensions {extensions} found in {folder}"
            )

        # Get frame counts for each video file
        self.frame_counts = []
        self.video_files = []

        for file in video_files:
            cap = cv2.VideoCapture(str(file))
            if not cap.isOpened():
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if frame_count <= 0:
                continue
            self.video_files.append(file)
            self.frame_counts.append(frame_count)

        if len(self.frame_counts) == 0:
            raise ValueError("No valid video files found with positive frame count")

        # Calculate selection probabilities based on frame counts
        total_frames = sum(self.frame_counts)
        self.probabilities = [count / total_frames for count in self.frame_counts]

        # Declare state variables
        self.current_video: cv2.VideoCapture
        self.current_video_index: int
        self.frames_read = 0
        self._select_new_video()

    def __del__(self) -> None:
        """Release resources when the generator is deleted."""
        if hasattr(self, "current_video") and self.current_video.isOpened():
            self.current_video.release()

    def __call__(self) -> torch.Tensor:
        """Generate a random image frame from the video files.

        Returns:
            Tensor representation of the image frame in RGB format with shape [C, H, W]

        Raises:
            RuntimeError: If frames cannot be read from the video
        """
        # Check if we need to select a new video
        if self._need_new_video():
            self._select_new_video()

        # Read frame
        ret, frame = self.current_video.read()

        # If reached end of video, select a new one
        if not ret:
            self._select_new_video()
            ret, frame = self.current_video.read()
            if not ret:
                raise RuntimeError("Failed to read frame from newly selected video")

        self.frames_read += 1

        # Convert frame to RGB and tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()

        return frame_tensor

    def _need_new_video(self) -> bool:
        """Determine if a new video needs to be selected.

        Returns:
            True if a new video needs to be selected, False otherwise
        """
        if (
            self.max_frames_per_video is not None
            and self.frames_read >= self.max_frames_per_video
        ):
            return True

        return False

    def _select_new_video(self) -> None:
        """Select a new video based on the calculated probabilities.

        Raises:
            RuntimeError: If the selected video file cannot be opened
        """
        # Close current video if open
        if hasattr(self, "current_video"):
            self.current_video.release()

        # Select a new video based on probabilities
        self.current_video_index = random.choices(
            range(len(self.video_files)), weights=self.probabilities, k=1
        )[0]

        video_path = self.video_files[self.current_video_index]
        self.current_video = cv2.VideoCapture(str(video_path))

        if not self.current_video.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        # Randomly position within the video
        total_frames = self.frame_counts[self.current_video_index]
        if total_frames > 1:  # More than one frame
            random_position = random.randint(0, total_frames - 1)
            self.current_video.set(cv2.CAP_PROP_POS_FRAMES, random_position)

        self.frames_read = 0
