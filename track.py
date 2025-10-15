# from ultralytics import YOLO
# import matplotlib.pyplot as plt

# # Load an official or custom model
# model = YOLO("best.pt")  # Load an official Detect model

# # Perform tracking with the model
# results = model.track(r"E:\Muqadas\projects\plant diseases prediction\YOLO\code\5243389-hd_1920_1080_25fps.mp4", show=True) 


# for result in results:
#     print(result.boxes)  # Boxes object for bbox outputs
#     print(result.masks)  # Masks object for segm outputs
#     print(result.keypoints)  # Keypoints object for keypoint outputs

#     print(result.probs)  # Class probabilities for cls outputs

# plt.show()


import cv2

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("best.pt")

# Open the video file
video_path = r"apple.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()