#include <cmath>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"

namespace mediapipe
{

namespace
{
constexpr char normalizedLandmarkListTag[] = "NORM_LANDMARKS";
constexpr char recognizedHandGestureTag[] = "RECOGNIZED_HAND_GESTURE";
} // namespace

// Graph config:
//
// node {
//   calculator: "HandGestureRecognitionCalculator"
//   input_stream: "NORM_LANDMARKS:scaled_landmarks"
// }
class HandGestureRecognitionCalculator : public CalculatorBase
{
public:
    static ::mediapipe::Status GetContract(CalculatorContract *cc);
    ::mediapipe::Status Open(CalculatorContext *cc) override;

    ::mediapipe::Status Process(CalculatorContext *cc) override;

private:
    float get_Euclidean_DistanceAB(float a_x, float a_y, float b_x, float b_y)
    {
        float dist = std::pow(a_x - b_x, 2) + pow(a_y - b_y, 2);
        return std::sqrt(dist);
    }

    bool isThumbNearFirstFinger(NormalizedLandmark point1, NormalizedLandmark point2)
    {
        float distance = this->get_Euclidean_DistanceAB(point1.x(), point1.y(), point2.x(), point2.y());
        return distance < 0.1;
    }

    enum LANDMARKS {
        TOP_THUMB = 4,
        BOTTOM_THUMB = 2,
        TOP_INDEX = 8,
        BOTTOM_INDEX = 6,
        TOP_MIDDLE = 12,
        BOTTOM_MIDDLE = 10,
        TOP_RING = 16,
        BOTTOM_RING = 14,
        TOP_PINKY = 20,
        BOTTOM_PINKY = 18,
        INDEX_KNUCK = 5,
        PINKY_KNUCK = 17,
        BOTTOM_PALM = 0
    };
};

REGISTER_CALCULATOR(HandGestureRecognitionCalculator);

::mediapipe::Status HandGestureRecognitionCalculator::GetContract(
    CalculatorContract *cc)
{
    RET_CHECK(cc->Inputs().HasTag(normalizedLandmarkListTag));
    cc->Inputs().Tag(normalizedLandmarkListTag).Set<mediapipe::NormalizedLandmarkList>();

    RET_CHECK(cc->Inputs().HasTag(handednessTag));
    cc->Inputs().Tag(handednessTag).Set<ClassificationList>();

    RET_CHECK(cc->Outputs().HasTag(recognizedHandGestureTag));
    cc->Outputs().Tag(recognizedHandGestureTag).Set<std::string>();

    return ::mediapipe::OkStatus();
}

::mediapipe::Status HandGestureRecognitionCalculator::Open(
    CalculatorContext *cc)
{
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
}

::mediapipe::Status HandGestureRecognitionCalculator::Process(
    CalculatorContext *cc)
{
    std::string *recognized_hand_gesture;

    const auto &landmarkList = cc->Inputs()
                                   .Tag(normalizedLandmarkListTag)
                                   .Get<mediapipe::NormalizedLandmarkList>();
    RET_CHECK_GT(landmarkList.landmark_size(), 0) << "Input landmark vector is empty.";

    // finger states
    bool thumbIsOpen = false;
    bool firstFingerIsOpen = false;
    bool secondFingerIsOpen = false;
    bool thirdFingerIsOpen = false;
    bool fourthFingerIsOpen = false;

    const auto thumbSide = (landmarkList.landmark(BOTTOM_INDEX).x() < landmarkList.landmark(BOTTOM_RING).x()) ? "Left" : "Right";


    float pseudoFixKeyPoint = landmarkList.landmark(TOP_THUMB).x();
    if ((thumbSide == "Left" && pseudoFixKeyPoint < landmarkList.landmark(BOTTOM_THUMB).x()) ||
        (thumbSide == "Right" && pseudoFixKeyPoint > landmarkList.landmark(BOTTOM_THUMB).x()))
    {
        thumbIsOpen = true;
    }

    pseudoFixKeyPoint = landmarkList.landmark(TOP_INDEX).y();
    if (pseudoFixKeyPoint < landmarkList.landmark(BOTTOM_INDEX).y())
    {
        firstFingerIsOpen = true;
    }

    pseudoFixKeyPoint = landmarkList.landmark(TOP_MIDDLE).y();
    if (pseudoFixKeyPoint < landmarkList.landmark(BOTTOM_MIDDLE).y())
    {
        secondFingerIsOpen = true;
    }

    pseudoFixKeyPoint = landmarkList.landmark(TOP_RING).y();
    if (pseudoFixKeyPoint < landmarkList.landmark(BOTTOM_RING).y())
    {
        thirdFingerIsOpen = true;
    }

    pseudoFixKeyPoint = landmarkList.landmark(TOP_PINKY).y();
    if (pseudoFixKeyPoint < landmarkList.landmark(BOTTOM_PINKY).y())
    {
        fourthFingerIsOpen = true;
    }

    // Hand gesture recognition
    if (thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && thirdFingerIsOpen && fourthFingerIsOpen)
    {
        recognized_hand_gesture = new std::string("FIVE");
    }
    else if (!thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && thirdFingerIsOpen && fourthFingerIsOpen)
    {
        recognized_hand_gesture = new std::string("FOUR");
    }
    else if ((thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen) ||
            (!thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && thirdFingerIsOpen && !fourthFingerIsOpen))
    {
        recognized_hand_gesture = new std::string("THREE");
    }
    else if (thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        recognized_hand_gesture = new std::string("TWO");
    }
    else if (!thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        recognized_hand_gesture = new std::string("ONE");
    }
    else if (!thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        recognized_hand_gesture = new std::string("YEAH");
    }
    else if (!thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && fourthFingerIsOpen)
    {
        recognized_hand_gesture = new std::string("ROCK");
    }
    else if (thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && fourthFingerIsOpen)
    {
        recognized_hand_gesture = new std::string("SPIDERMAN");
    }
    else if (!thumbIsOpen && !firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        recognized_hand_gesture = new std::string("FIST");
    }
    else if (!firstFingerIsOpen && secondFingerIsOpen && thirdFingerIsOpen && fourthFingerIsOpen && this->isThumbNearFirstFinger(landmarkList.landmark(TOP_THUMB), landmarkList.landmark(TOP_INDEX)))
    {
        recognized_hand_gesture = new std::string("OK");
    }
    else if (!firstFingerIsOpen && secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        recognized_hand_gesture = new std::string("BIRD");
    }
    else if (thumbIsOpen && !firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && fourthFingerIsOpen)
    {
        recognized_hand_gesture = new std::string("SHAKA");
    }
    else
    {
        recognized_hand_gesture = new std::string("NONE");
        //LOG(INFO) << "Finger States: " << thumbIsOpen << firstFingerIsOpen << secondFingerIsOpen << thirdFingerIsOpen << fourthFingerIsOpen;       
    }
    //LOG(INFO) << "Finger States: " << thumbIsOpen << firstFingerIsOpen << secondFingerIsOpen << thirdFingerIsOpen << fourthFingerIsOpen;       

    //LOG(INFO) << recognized_hand_gesture;

    //LOG(INFO) << "Palm Dir: " << palmFacing << ", hand: " << handedness;

    cc->Outputs()
        .Tag(recognizedHandGestureTag)
        .Add(recognized_hand_gesture, cc->InputTimestamp());

    return ::mediapipe::OkStatus();
} // namespace mediapipe

} // namespace mediapipe
