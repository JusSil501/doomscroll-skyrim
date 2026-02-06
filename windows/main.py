import cv2  
import mediapipe as mp  
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import threading
import urllib.request
import os
from pathlib import Path


class VideoPlayer:
    """Cross-platform video player using OpenCV."""
    
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.playing = False
        self.thread = None
        self.cap = None
        self.window_name = "Skyrim Alert"
    
    def _play_loop(self):
        """Internal loop that plays the video in a separate thread."""
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            print(f"Could not open video: {self.video_path}")
            return
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        delay = int(1000 / fps)
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 390, 780)
        
        while self.playing:
            ret, frame = self.cap.read()
            if not ret:
                # Loop the video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            try:
                cv2.imshow(self.window_name, frame)
                key = cv2.waitKey(delay) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or Q to close
                    break
            except:
                break
        
        self.cap.release()
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass
    
    def play(self):
        """Start playing the video in a separate window."""
        if not self.playing:
            self.playing = True
            self.thread = threading.Thread(target=self._play_loop, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop playing the video."""
        self.playing = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None


# Global video player instance
video_player = None


def play_video(video_path: Path) -> None:
    """Play video in a separate window (cross-platform)."""
    global video_player
    if video_player is None:
        video_player = VideoPlayer(video_path)
    video_player.play()


def close_video(video_path: Path) -> None:
    """Stop the video playback."""
    global video_player
    if video_player is not None:
        video_player.stop()

def draw_warning(frame, text="lock in twin"):
    h, w = frame.shape[:2]
    box_w, box_h = 500, 70
    x1 = (w - box_w) // 2
    y1 = 24
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 0, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (80, 255, 160) , 4)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 160) , 2)

    cv2.putText(
        frame,
        text.upper(),
        (x1 + 26, y1 + 48),
        cv2.FONT_HERSHEY_DUPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )


    
def main():
    timer = 2.0
    looking_down_threshold = 0.25
    debounce_threshold = 0.45
    
    skyrim_skeleton_video = Path("./assets/skyrim-skeleton.mp4").resolve()
    if not skyrim_skeleton_video.exists():
        print("Could not open skyrim-skeleton.mp4")
        return
    
    # Download the face landmarker model if it doesn't exist
    model_path = Path("./assets/face_landmarker.task").resolve()
    if not model_path.exists():
        print("Downloading face landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, str(model_path))
        print("Model downloaded successfully!")
    
    # Create face landmarker using the new API
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Could not open webcam")
        return
    
    doomscroll = None
    video_playing = False

    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to mediapipe image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect face landmarks
        detection_result = face_landmarker.detect(mp_image)
        face_landmark_points = detection_result.face_landmarks

        current = time.time()

        if face_landmark_points and len(face_landmark_points) > 0:
            one_face_landmark_points = face_landmark_points[0]
            
            left = [one_face_landmark_points[145], one_face_landmark_points[159]]
            for landmark_point in left:
                x = int(landmark_point.x * width)
                y = int(landmark_point.y * height)

            right = [one_face_landmark_points[374], one_face_landmark_points[386]]
            for landmark_point in right:
                    x = int(landmark_point.x * width)
                    y = int(landmark_point.y * height)
                 
            
            lx = int((left[0].x + left[1].x) / 2 * width)
            ly = int((left[0].y + left[1].y) / 2 * height)

            rx = int((right[0].x + right[1].x) / 2 * width)
            ry = int((right[0].y + right[1].y) / 2 * height)

            box = 50

            cv2.rectangle(frame, (lx - box, ly - box), (lx + box, ly + box), (10, 255, 0), 2)
            cv2.rectangle(frame, (rx - box, ry - box), (rx + box, ry + box), (10, 255, 0), 2)
            

            l_iris = one_face_landmark_points[468]
            r_iris = one_face_landmark_points[473]
            
            l_ratio = (l_iris.y  - left[1].y)  / (left[0].y  - left[1].y  + 1e-6)
            r_ratio = (r_iris.y - right[1].y) / (right[0].y - right[1].y + 1e-6)

            avg_ratio = (l_ratio + r_ratio) / 2.0

            if video_playing:
                is_looking_down = avg_ratio < debounce_threshold
            else:
                is_looking_down = avg_ratio < looking_down_threshold


            if is_looking_down:
                if doomscroll is None:
                    doomscroll = current

                if (current - doomscroll) >= timer:               
                    if not video_playing:
                        play_video(skyrim_skeleton_video)
                        video_playing = True

            else:
                doomscroll = None
                if video_playing:
                    close_video(skyrim_skeleton_video)
                    video_playing = False
        else:
            doomscroll = None
            if video_playing:
                close_video(skyrim_skeleton_video)
                video_playing = False

        if video_playing:
            draw_warning(frame, "doomscrolling alarm")

        cv2.imshow('lock in', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):
            break

    if video_playing:
        close_video(skyrim_skeleton_video)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


