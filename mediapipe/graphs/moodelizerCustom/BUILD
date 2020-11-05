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

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mobile_calculators",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//input-model-gate:input-model-gate-calculator",
    ],
)

cc_library(
    name = "face_mesh_calculators",
    deps = [
        "//mediapipe/modules/face_landmark:face_landmark_front_gpu",
    ],
)

cc_library(
    name = "hand_tracking_calculators",
    deps = [
        "//mediapipe/modules/face_landmark:face_landmark_front_gpu",
        "//hand-gesture-recognition:hand-gesture-recognition-calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:merge_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/graphs/hand_tracking/subgraphs:hand_detection_gpu",
        "//mediapipe/graphs/hand_tracking/subgraphs:hand_landmark_gpu",
    ],
)

mediapipe_binary_graph(
    name = "moodelizer_custom_binary_graph",
    graph = "moodelizer_custom_graph.pbtxt",
    output_name = "moodelizer_custom_graph.binarypb",
    deps = [":mobile_calculators", ":face_mesh_calculators", ":hand_tracking_calculators"],
)
