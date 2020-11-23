---
layout: default
title: Home
nav_order: 1
---

![MediaPipe](docs/images/mediapipe_small.png)

--------------------------------------------------------------------------------
## Generate mediapipe Android Archive File (AAR) and detection binary file

1. Fork this repository
2. [Install dependencies](https://google.github.io/mediapipe/getting_started/install.html#installing-on-macos), make sure that tools/framework such as Bazel, OpenCV, FFmpeg are installed first
3. Install Android Development SDK and Android NDK, if Android SDK and NDK are already installed (e.g., by Android Studio), set $ANDROID_HOME and $ANDROID_NDK_HOME to point to the installed SDK and NDK.
  * `export ANDROID_HOME=<path to the Android SDK>`
  * `export ANDROID_NDK_HOME=<path to the Android NDK>`
4. Let's say you would like to modify [pose tracking graph](https://github.com/Moodelizer/mediapipe/blob/development/mediapipe/graphs/pose_tracking/upper_body_pose_tracking_custom.pbtxt) to change the smoothing calculator velocity_scale from default value 10.0 to 15.0 to see if it affects
the tracking performance. You'd make the change and start building the AAR. Note that you could always visulize the graph in the [web browser](https://viz.mediapipe.dev/)
5. Run the Bazel build command to generate the AAR
  * Our custom BUILD file was a modification from handdetectiongpu example Build, you could find it [here] (https://github.com/Moodelizer/mediapipe/blob/development/mediapipe/examples/android/src/java/com/google/mediapipe/apps/handdetectiongpu/BUILD), in order to build the final AAR, following the steps below:
  * Run `cd path_to_mediapipe_repo_root/mediapipe/examples/android/src/java/com/google/mediapipe/apps` to go to right folder
  * Run `bazel build -c opt --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --linkopt="-s" --fat_apk_cpu=arm64-v8a,armeabi-v7a handdetectiongpu:mp_multi_input_aar`
  * Locate generated AAR file in `ls bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/handdetectiongpu/mp_multi_input_aar.aar`
  * Copy the AAR into app's Anroid Library `mp_multi_input_aar` in the Android project.
6. Build the relevant binary graph and copy the assets into `app/src/main/assets`
  * Run `cd path_to_mediapipe_repo_root/`
  * Run `bazel build -c opt mediapipe/graphs/pose_tracking:mobile_upper_body_pose_tracking_gpu_binary_graph`
  * Locate generated binary graph file in `ls bazel-bin/mediapipe/graphs/pose_tracking/mobile_upper_body_pose_tracking_gpu_binary_graph.binarypb`
7. (Optional) When adding a new detection, often it'd also need to copy the tflite model into Android Studio project's assets folder.
8. Now the output stream from mediapipe should be up for grab inside of Android Studio, time to build Android App!

## Detections that are used in Live app

There are several graphs that were involved in generating the `mp_multi_input_aar.aar` for the usage of our Live App.
* Consolidated graph for [Face Detections and Face Effect](https://github.com/Moodelizer/mediapipe/blob/development/mediapipe/graphs/face_custom/face_consolidated.pbtxt)
* [Face Mesh](https://github.com/Moodelizer/mediapipe/blob/development/mediapipe/graphs/face_mesh/face_mesh_custom.pbtxt)
* [Hand Tracking](https://github.com/Moodelizer/mediapipe/blob/development/mediapipe/graphs/hand_tracking/hand_tracking_custom.pbtxt)
* [Pose Tracking](https://github.com/Moodelizer/mediapipe/blob/development/mediapipe/graphs/pose_tracking/upper_body_pose_tracking_custom.pbtxt)
There are two custom calculators that were located in the folder called `hand-gesture-recognition` which is under the project root folder, they were severed to [recognize the gesture](https://github.com/Moodelizer/mediapipe/blob/development/hand-gesture-recognition/hand-gesture-recognition-calculator.cc) and [projecting the landmasrks](https://github.com/Moodelizer/mediapipe/blob/development/hand-gesture-recognition/custom-landmark-projection-calculator.cc), the hand gesture recognition calculator was a modified version of [this](https://gist.github.com/TheJLifeX/74958cc59db477a91837244ff598ef4a)

## Live ML anywhere

[MediaPipe](https://google.github.io/mediapipe/) offers cross-platform, customizable
ML solutions for live and streaming media.

![accelerated.png](docs/images/accelerated_small.png)                                                               | ![cross_platform.png](docs/images/cross_platform_small.png)
:------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------:
***End-to-End acceleration***: *Built-in fast ML inference and processing accelerated even on common hardware* | ***Build once, deploy anywhere***: *Unified solution works across Android, iOS, desktop/cloud, web and IoT*
![ready_to_use.png](docs/images/ready_to_use_small.png)                                                             | ![open_source.png](docs/images/open_source_small.png)
***Ready-to-use solutions***: *Cutting-edge ML solutions demonstrating full power of the framework*            | ***Free and open source***: *Framework and solutions both under Apache 2.0, fully extensible and customizable*

## ML solutions in MediaPipe

Face Detection                                                                                                                 | Face Mesh                                                                                                       | Iris                                                                                                      | Hands                                                                                                      | Pose                                                                                                      | Hair Segmentation
:----------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------: | :---------------:
[![face_detection](docs/images/mobile/face_detection_android_gpu_small.gif)](https://google.github.io/mediapipe/solutions/face_detection) | [![face_mesh](docs/images/mobile/face_mesh_android_gpu_small.gif)](https://google.github.io/mediapipe/solutions/face_mesh) | [![iris](docs/images/mobile/iris_tracking_android_gpu_small.gif)](https://google.github.io/mediapipe/solutions/iris) | [![hand](docs/images/mobile/hand_tracking_android_gpu_small.gif)](https://google.github.io/mediapipe/solutions/hands) | [![pose](docs/images/mobile/pose_tracking_android_gpu_small.gif)](https://google.github.io/mediapipe/solutions/pose) | [![hair_segmentation](docs/images/mobile/hair_segmentation_android_gpu_small.gif)](https://google.github.io/mediapipe/solutions/hair_segmentation)

Object Detection                                                                                                                     | Box Tracking                                                                                                                | Instant Motion Tracking                                                                                                                               | Objectron                                                                                                             | KNIFT
:----------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------: | :---:
[![object_detection](docs/images/mobile/object_detection_android_gpu_small.gif)](https://google.github.io/mediapipe/solutions/object_detection) | [![box_tracking](docs/images/mobile/object_tracking_android_gpu_small.gif)](https://google.github.io/mediapipe/solutions/box_tracking) | [![instant_motion_tracking](docs/images/mobile/instant_motion_tracking_android_small.gif)](https://google.github.io/mediapipe/solutions/instant_motion_tracking) | [![objectron](docs/images/mobile/objectron_chair_android_gpu_small.gif)](https://google.github.io/mediapipe/solutions/objectron) | [![knift](docs/images/mobile/template_matching_android_cpu_small.gif)](https://google.github.io/mediapipe/solutions/knift)

<!-- []() in the first cell is needed to preserve table formatting in GitHub Pages. -->
<!-- Whenever this table is updated, paste a copy to solutions/solutions.md. -->

[]()                                                                                      | Android | iOS | Desktop | Python | Web | Coral
:---------------------------------------------------------------------------------------- | :-----: | :-: | :-----: | :----: | :-: | :---:
[Face Detection](https://google.github.io/mediapipe/solutions/face_detection)                   | ✅       | ✅   | ✅       |        | ✅   | ✅
[Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh)                             | ✅       | ✅   | ✅       | ✅      |     |
[Iris](https://google.github.io/mediapipe/solutions/iris)                                       | ✅       | ✅   | ✅       |        | ✅   |
[Hands](https://google.github.io/mediapipe/solutions/hands)                                     | ✅       | ✅   | ✅       | ✅      | ✅   |
[Pose](https://google.github.io/mediapipe/solutions/pose)                                       | ✅       | ✅   | ✅       | ✅      | ✅   |
[Hair Segmentation](https://google.github.io/mediapipe/solutions/hair_segmentation)             | ✅       |     | ✅       |        | ✅   |
[Object Detection](https://google.github.io/mediapipe/solutions/object_detection)               | ✅       | ✅   | ✅       |        |     | ✅
[Box Tracking](https://google.github.io/mediapipe/solutions/box_tracking)                       | ✅       | ✅   | ✅       |        |     |
[Instant Motion Tracking](https://google.github.io/mediapipe/solutions/instant_motion_tracking) | ✅       |     |         |        |     |
[Objectron](https://google.github.io/mediapipe/solutions/objectron)                             | ✅       |     |         |        |     |
[KNIFT](https://google.github.io/mediapipe/solutions/knift)                                     | ✅       |     |         |        |     |
[AutoFlip](https://google.github.io/mediapipe/solutions/autoflip)                               |         |     | ✅       |        |     |
[MediaSequence](https://google.github.io/mediapipe/solutions/media_sequence)                    |         |     | ✅       |        |     |
[YouTube 8M](https://google.github.io/mediapipe/solutions/youtube_8m)                           |         |     | ✅       |        |     |

See also
[MediaPipe Models and Model Cards](https://google.github.io/mediapipe/solutions/models)
for ML models released in MediaPipe.

## MediaPipe in Python

MediaPipe Python package is available on
[PyPI](https://pypi.org/project/mediapipe/), and can be installed simply by `pip
install mediapipe` on Linux and macOS, as described in:

*   [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh#python)
    ([colab](https://mediapipe.page.link/face_mesh_py_colab))
*   [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands#python)
    ([colab](https://mediapipe.page.link/hands_py_colab))
*   [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose#python)
    ([colab](https://mediapipe.page.link/pose_py_colab))

## MediaPipe on the Web

MediaPipe on the Web is an effort to run the same ML solutions built for mobile
and desktop also in web browsers. The official API is under construction, but
the core technology has been proven effective. Please see
[MediaPipe on the Web](https://developers.googleblog.com/2020/01/mediapipe-on-web.html)
in Google Developers Blog for details.

You can use the following links to load a demo in the MediaPipe Visualizer, and
over there click the "Runner" icon in the top bar like shown below. The demos
use your webcam video as input, which is processed all locally in real-time and
never leaves your device.

![visualizer_runner](docs/images/visualizer_runner.png)

*   [MediaPipe Face Detection](https://viz.mediapipe.dev/demo/face_detection)
*   [MediaPipe Iris](https://viz.mediapipe.dev/demo/iris_tracking)
*   [MediaPipe Iris: Depth-from-Iris](https://viz.mediapipe.dev/demo/iris_depth)
*   [MediaPipe Hands](https://viz.mediapipe.dev/demo/hand_tracking)
*   [MediaPipe Hands (palm/hand detection only)](https://viz.mediapipe.dev/demo/hand_detection)
*   [MediaPipe Pose](https://viz.mediapipe.dev/demo/pose_tracking)
*   [MediaPipe Hair Segmentation](https://viz.mediapipe.dev/demo/hair_segmentation)

## Getting started

Learn how to [install](https://google.github.io/mediapipe/getting_started/install)
MediaPipe and
[build example applications](https://google.github.io/mediapipe/getting_started/building_examples),
and start exploring our ready-to-use
[solutions](https://google.github.io/mediapipe/solutions/solutions) that you can
further extend and customize.

The source code is hosted in the
[MediaPipe Github repository](https://github.com/google/mediapipe), and you can
run code search using
[Google Open Source Code Search](https://cs.opensource.google/mediapipe/mediapipe).

## Publications

*   [MediaPipe 3D Face Transform](https://developers.googleblog.com/2020/09/mediapipe-3d-face-transform.html)
    in Google Developers Blog
*   [Instant Motion Tracking With MediaPipe](https://developers.googleblog.com/2020/08/instant-motion-tracking-with-mediapipe.html)
    in Google Developers Blog
*   [BlazePose - On-device Real-time Body Pose Tracking](https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html)
    in Google AI Blog
*   [MediaPipe Iris: Real-time Eye Tracking and Depth Estimation](https://ai.googleblog.com/2020/08/mediapipe-iris-real-time-iris-tracking.html)
    in Google AI Blog
*   [MediaPipe KNIFT: Template-based feature matching](https://developers.googleblog.com/2020/04/mediapipe-knift-template-based-feature-matching.html)
    in Google Developers Blog
*   [Alfred Camera: Smart camera features using MediaPipe](https://developers.googleblog.com/2020/03/alfred-camera-smart-camera-features-using-mediapipe.html)
    in Google Developers Blog
*   [Real-Time 3D Object Detection on Mobile Devices with MediaPipe](https://ai.googleblog.com/2020/03/real-time-3d-object-detection-on-mobile.html)
    in Google AI Blog
*   [AutoFlip: An Open Source Framework for Intelligent Video Reframing](https://ai.googleblog.com/2020/02/autoflip-open-source-framework-for.html)
    in Google AI Blog
*   [MediaPipe on the Web](https://developers.googleblog.com/2020/01/mediapipe-on-web.html)
    in Google Developers Blog
*   [Object Detection and Tracking using MediaPipe](https://developers.googleblog.com/2019/12/object-detection-and-tracking-using-mediapipe.html)
    in Google Developers Blog
*   [On-Device, Real-Time Hand Tracking with MediaPipe](https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html)
    in Google AI Blog
*   [MediaPipe: A Framework for Building Perception Pipelines](https://arxiv.org/abs/1906.08172)

## Videos

*   [YouTube Channel](https://www.youtube.com/c/MediaPipe)

## Events

*   [MediaPipe Seattle Meetup, Google Building Waterside, 13 Feb 2020](https://mediapipe.page.link/seattle2020)
*   [AI Nextcon 2020, 12-16 Feb 2020, Seattle](http://aisea20.xnextcon.com/)
*   [MediaPipe Madrid Meetup, 16 Dec 2019](https://www.meetup.com/Madrid-AI-Developers-Group/events/266329088/)
*   [MediaPipe London Meetup, Google 123 Building, 12 Dec 2019](https://www.meetup.com/London-AI-Tech-Talk/events/266329038)
*   [ML Conference, Berlin, 11 Dec 2019](https://mlconference.ai/machine-learning-advanced-development/mediapipe-building-real-time-cross-platform-mobile-web-edge-desktop-video-audio-ml-pipelines/)
*   [MediaPipe Berlin Meetup, Google Berlin, 11 Dec 2019](https://www.meetup.com/Berlin-AI-Tech-Talk/events/266328794/)
*   [The 3rd Workshop on YouTube-8M Large Scale Video Understanding Workshop,
    Seoul, Korea ICCV
    2019](https://research.google.com/youtube8m/workshop2019/index.html)
*   [AI DevWorld 2019, 10 Oct 2019, San Jose, CA](https://aidevworld.com)
*   [Google Industry Workshop at ICIP 2019, 24 Sept 2019, Taipei, Taiwan](http://2019.ieeeicip.org/?action=page4&id=14#Google)
    ([presentation](https://docs.google.com/presentation/d/e/2PACX-1vRIBBbO_LO9v2YmvbHHEt1cwyqH6EjDxiILjuT0foXy1E7g6uyh4CesB2DkkEwlRDO9_lWfuKMZx98T/pub?start=false&loop=false&delayms=3000&slide=id.g556cc1a659_0_5))
*   [Open sourced at CVPR 2019, 17~20 June, Long Beach, CA](https://sites.google.com/corp/view/perception-cv4arvr/mediapipe)

## Community

*   [Awesome MediaPipe](https://mediapipe.org) - A curated list of awesome
    MediaPipe related frameworks, libraries and software
*   [Slack community](https://mediapipe.page.link/joinslack) for MediaPipe users
*   [Discuss](https://groups.google.com/forum/#!forum/mediapipe) - General
    community discussion around MediaPipe

## Alpha disclaimer

MediaPipe is currently in alpha at v0.7. We may be still making breaking API
changes and expect to get to stable APIs by v1.0.

## Contributing

We welcome contributions. Please follow these
[guidelines](https://github.com/google/mediapipe/blob/master/CONTRIBUTING.md).

We use GitHub issues for tracking requests and bugs. Please post questions to
the MediaPipe Stack Overflow with a `mediapipe` tag.
