import numpy as np
import cv2
import os
import mediapipe as mp

class FilmGeneratorLifesPhoto():
    def __init__(self):
        self.model = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                                refine_landmarks=True,
                                                min_detection_confidence=0.5,
                                                min_tracking_confidence=0.5)
        self.img_dir = "res"
        self.delta_x_std = 20
        self.frame_rate = 1
        self.frame_size = (640, 480)
        self.bg_width = self.frame_size[0]
        self.bg_height = self.frame_size[1]
        self.video_writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), self.frame_rate, self.frame_size)


    def release_video(self):
        img_files = os.listdir(self.img_dir)
        img_files.sort()

        for image in img_files:
            img_file = os.path.join(self.img_dir, image)
            img = cv2.imread(img_file)
            img_height, img_width, _ = img.shape
            
            eyes_loc = self.get_eyes_loc(img, img_height, img_width)
            if eyes_loc is not None and len(eyes_loc) == 2:
                delta_x = abs(eyes_loc[1][0] - eyes_loc[0][0])

            white_image = np.full((self.bg_height, self.bg_width, 3), 200, dtype=np.uint8)
            offset_x = (self.bg_width - img_width) // 2
            offset_y = (self.bg_height - img_height) // 2
            white_image[offset_y:offset_y+img_height, offset_x:offset_x+img_width] = img
            
            self.video_writer.write(white_image)

        self.video_writer.release()
        cv2.destroyAllWindows()

    def get_eyes_loc(self, img, img_height, img_width):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lms = self.model.process(img_rgb).multi_face_landmarks
        landmark_index = 0
        eyes_loc = []
        if lms:
            for lm in lms:
                for point in lm.landmark:
                    if landmark_index in (463, 243):
                        lm0 = point.x, point.y
                        eyes_loc.append((int(lm0[0]*img_width), int(lm0[1]*img_height)))
                        cv2.circle(img, (int(lm0[0]*img_width), int(lm0[1]*img_height)), 3, (0, 0, 255), cv2.FILLED)
                    landmark_index += 1
        return eyes_loc

fg = FilmGeneratorLifesPhoto()
fg.release_video()