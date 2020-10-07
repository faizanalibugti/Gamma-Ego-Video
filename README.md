# Gamma + Ego Lane Segmentation on Video

This code will allow inference of gamma + ego lane segmentation model on video data

You must have moviepy library installed 

1. Download or clone this repository
2. Rename your video as **drive.mp4**
3. In Anaconda Prompt, run **python night.py**, this will generate the final output night.mp4

It may prompt with the error:

cv2.error: OpenCV(3.4.2) C:\projects\opencv-python\opencv\modules\core\src\arithm.cpp:659: error: (-209:Sizes of input arguments do not match) The operation is neither 'array op array' (where arrays have the same size and the same number of channels), nor 'array op scalar', nor 'scalar op array' in function 'cv::arithm_op'

To resolve this modify **Line 44** night.py to input image resolution of video

You will find correct resolution printed on Anaconda Prompt when night.py is run

Final output will be **night.mp4**