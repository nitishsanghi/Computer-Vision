# Vehicle Detection

In this project, a software pipeline is written to detect vehicles in a video. To accomplish the goal of the project a number of computer vision concepts are applied couple with a machine learning technique. A Histogram of Oriented Gradients (HOG) based feature extraction is performed on a labeled training set of images and a Linear SVM classifier is trained using those features. Sliding-window technique is couple with the trained classifier to search for vehicles in image frames of the video. The pipeline is run on a video stream to create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles. Finally, an estimated bounding box is created for the vehicles detected. The resulting output stream and the pipeline are discussed below.

![Alt Text](https://media.giphy.com/media/N9IzYguViAsUTLyuGV/giphy-downsized-large.gif)

<img src="https://github.com/nitishsanghi/Computer-Vision/blob/main/VehicleDetection/pipelineimages/step1.png" width="750" height="250">
<img src="https://github.com/nitishsanghi/Computer-Vision/blob/main/VehicleDetection/pipelineimages/step2.png" width="750" height="250">
<img src="https://github.com/nitishsanghi/Computer-Vision/blob/main/VehicleDetection/pipelineimages/step3.png" width="750" height="250">
<img src="https://github.com/nitishsanghi/Computer-Vision/blob/main/VehicleDetection/pipelineimages/step4.png" width="750" height="250">
<img src="https://github.com/nitishsanghi/Computer-Vision/blob/main/VehicleDetection/pipelineimages/step5.png" width="750" height="250">
<img src="https://github.com/nitishsanghi/Computer-Vision/blob/main/VehicleDetection/pipelineimages/step6.png" width="750" height="250">
<img src="https://github.com/nitishsanghi/Computer-Vision/blob/main/VehicleDetection/pipelineimages/step7.png" width="750" height="250">
