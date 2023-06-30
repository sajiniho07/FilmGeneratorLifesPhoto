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
        self.left_eye_loc_std = [300, 150]
        self.frame_rate = 1
        self.frame_size = (640, 480)
        self.bg_width = self.frame_size[0]
        self.bg_height = self.frame_size[1]
        self.video_writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), self.frame_rate, self.frame_size)


    def release_video(self):
        img_files = sorted(os.listdir(self.img_dir))

        for img_name in img_files:
            img_file = os.path.join(self.img_dir, img_name)
            img = cv2.imread(img_file)
            img_height, img_width, _ = img.shape

            eyes_loc = self.get_eyes_loc(img, img_height, img_width)
            offset_x = self.left_eye_loc_std[0] - eyes_loc[0][0]
            offset_y = self.left_eye_loc_std[1] - eyes_loc[0][1]

            resized_image, offset_x, offset_y = self.resize_image(img, offset_x, offset_y)
            
            img_height, img_width, _ = resized_image.shape
            new_width = offset_x + img_width
            new_height = offset_y + img_height
            
            try:
                white_image = np.full((self.bg_height, self.bg_width, 3), 200, dtype=np.uint8)
                white_image[offset_y:new_height, offset_x:new_width] = resized_image
                self.video_writer.write(white_image)
            except:
                print(f'Couldnt write this file: {img_name}')
                pass

        self.video_writer.release()
        cv2.destroyAllWindows()

    def resize_image(self, img, offset_x, offset_y):
        img_height, img_width, _ = img.shape
        new_width = offset_x + img_width
        new_height = offset_y + img_height

        if new_height > self.bg_height and new_width > self.bg_width:
            diff_w = new_width - self.bg_width
            diff_h = new_height - self.bg_height
            new_size = (img_width - diff_w, img_height - diff_h)
            resized_image = cv2.resize(img, new_size)

        elif new_width > self.bg_width:
            diff_w = new_width - self.bg_width
            new_size = (img_width - diff_w, img_height - diff_w)
            resized_image = cv2.resize(img, new_size)

        elif new_height > self.bg_height:
            diff_h = new_height - self.bg_height
            new_size = (img_width - diff_h, img_height - diff_h)
            resized_image = cv2.resize(img, new_size)

        else:
            resized_image = img.copy()

        if offset_x < 0:
            resized_image = resized_image[:, abs(offset_x):img_width]
            offset_x = 0
        if offset_y < 0:
            resized_image = resized_image[abs(offset_y):img_height, :]
            offset_y = 0

        return resized_image, offset_x, offset_y
    
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
                        # cv2.circle(img, (int(lm0[0]*img_width), int(lm0[1]*img_height)), 3, (0, 0, 255), cv2.FILLED)
                    landmark_index += 1
        return eyes_loc

fg = FilmGeneratorLifesPhoto()
fg.release_video()