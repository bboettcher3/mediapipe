# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mobile_calculators",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/modules/face_landmark:face_landmark_front_gpu",
        "//mediapipe/modules/face_geometry",
        "//mediapipe/modules/face_geometry:effect_renderer_calculator",
        "//mediapipe/modules/face_geometry:env_generator_calculator",
        "//mediapipe/modules/face_detection:face_detection_front_cpu",
        "//mediapipe/modules/face_detection:face_detection_front_gpu",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "//mediapipe/gpu:image_frame_to_gpu_buffer_calculator",
    ],
)

mediapipe_binary_graph(
    name = "face_consolidated_mobile_gpu_binary_graph",
    graph = "face_consolidated.pbtxt",
    output_name = "face_consolidated_mobile_gpu.binarypb",
    deps = [":mobile_calculators"],
)
