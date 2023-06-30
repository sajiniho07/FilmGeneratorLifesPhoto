# Film Generator Life's Photo

Film Generator Life's Photo is a Python script that generates a video by composing images from a directory. The script uses the MediaPipe library to detect facial landmarks and aligns the images based on the location of the eyes.

## Prerequisites
- Python 3.x
- OpenCV (`cv2`) library
- NumPy library
- MediaPipe library

## Installation
1. Clone the repository or download the `film_generator_lifes_photo.py` file.
2. Install the required dependencies by running the following command:
   ```
   pip install opencv-python numpy mediapipe
   ```

## Usage
1. Place the images you want to include in the video in a directory.
2. Update the `img_dir` variable in the script with the path to the image directory.
3. Adjust the parameters according to your preferences:
   - `left_eye_loc_std`: Specifies the standard location of the left eye (x, y) for alignment. Adjust this if the alignment is incorrect.
   - `frame_rate`: Specifies the frame rate of the generated video.
   - `frame_size`: Specifies the frame size (width, height) of the generated video.
4. Run the script using the following command:
   ```
   python film_generator_lifes_photo.py
   ```
5. The script will process the images, align them based on the location of the eyes, and generate a video named `output.mp4` in the current directory.

Note: Make sure the images in the directory have consistent aspect ratios for better alignment results.

## Example 

![sample](output.gif)

## Acknowledgements
- [OpenCV](https://opencv.org/): Open Source Computer Vision Library
- [NumPy](https://numpy.org/): Fundamental package for scientific computing with Python
- [MediaPipe](https://mediapipe.dev/): Cross-platform framework for building multimodal applied machine learning pipelines

## License

Made with :heart: by <a href="https://github.com/sajiniho07" target="_blank">Sajad Kamali</a>

&#xa0;

<a href="#top">Back to top</a>
