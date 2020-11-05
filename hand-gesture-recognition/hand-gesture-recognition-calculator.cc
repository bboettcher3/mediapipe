#include <cmath>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"

namespace mediapipe
{

namespace
{
constexpr char normRectTag[] = "NORM_RECT";
constexpr char normalizedLandmarkListTag[] = "NORM_LANDMARKS";
constexpr char handednessTag[] = "CLASSIFICATIONS";
constexpr char recognizedHandGestureTag[] = "RECOGNIZED_HAND_GESTURE";
} // namespace

// Graph config:
//
// node {
//   calculator: "HandGestureRecognitionCalculator"
//   input_stream: "NORM_LANDMARKS:scaled_landmarks"
//   input_stream: "NORM_RECT:hand_rect_for_next_frame"
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

    bool isValidDetection(NormalizedLandmark knuckLeft, NormalizedLandmark knuckRight, NormalizedLandmark bottomPalm, NormalizedLandmark topThumb)
    {
        // Computes distance between opposite knuckles
        /* float knuckDist = this->get_Euclidean_DistanceAB(knuckLeft.x(), knuckLeft.y(), knuckRight.x(), knuckRight.y());
        // Computes distance between thumb nail and bottom of palm (pretty much physically impossible)
        float palmDist = this->get_Euclidean_DistanceAB(bottomPalm.x(), bottomPalm.y(), topThumb.x(), topThumb.y());
        if (knuckDist < 0.1 || palmDist < 0.28)
        {
            return false;
        }
        else
        {
            //LOG(INFO) << "Knuck: " << knuckDist << ", Palm: " << palmDist;
        } */
        return true;
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
    enum PALM_DIR {
        FRONT,
        BACK
    };
};

REGISTER_CALCULATOR(HandGestureRecognitionCalculator);

::mediapipe::Status HandGestureRecognitionCalculator::GetContract(
    CalculatorContract *cc)
{
    RET_CHECK(cc->Inputs().HasTag(normalizedLandmarkListTag));
    cc->Inputs().Tag(normalizedLandmarkListTag).Set<mediapipe::NormalizedLandmarkList>();

    RET_CHECK(cc->Inputs().HasTag(normRectTag));
    cc->Inputs().Tag(normRectTag).Set<NormalizedRect>();

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

    // hand closed (red) rectangle
    const auto rect = &(cc->Inputs().Tag(normRectTag).Get<NormalizedRect>());
    float width = rect->width();
    float height = rect->height();

    if (width < 0.01 || height < 0.01)
    {
        // LOG(INFO) << "No Hand Detected";
        recognized_hand_gesture = new std::string("NONE");
        cc->Outputs()
            .Tag(recognizedHandGestureTag)
            .Add(recognized_hand_gesture, cc->InputTimestamp());
        return ::mediapipe::OkStatus();
    }    

    //LOG(INFO) << "w: " << width << ", h: " << height;

    const auto &landmarkList = cc->Inputs()
                                   .Tag(normalizedLandmarkListTag)
                                   .Get<mediapipe::NormalizedLandmarkList>();
    RET_CHECK_GT(landmarkList.landmark_size(), 0) << "Input landmark vector is empty.";

    if (!this->isValidDetection(landmarkList.landmark(INDEX_KNUCK), landmarkList.landmark(PINKY_KNUCK), landmarkList.landmark(BOTTOM_PALM), landmarkList.landmark(TOP_THUMB)))
    {
        LOG(INFO) << "Bunched up points, discarding..";
        recognized_hand_gesture = new std::string("NONE");
        cc->Outputs()
            .Tag(recognizedHandGestureTag)
            .Add(recognized_hand_gesture, cc->InputTimestamp());
        return ::mediapipe::OkStatus();
    }

    // finger states
    bool thumbIsOpen = false;
    bool firstFingerIsOpen = false;
    bool secondFingerIsOpen = false;
    bool thirdFingerIsOpen = false;
    bool fourthFingerIsOpen = false;
    //

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
    else if (thumbIsOpen && !firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        bool isThumbAbove = landmarkList.landmark(BOTTOM_THUMB).x() < landmarkList.landmark(BOTTOM_PINKY).x();
        bool isLeftHand = rect->rotation() > 0;
        if (!(isThumbAbove ^ isLeftHand))
        {
            recognized_hand_gesture = new std::string("THUMBSUP");
        }
        else
        {
            recognized_hand_gesture = new std::string("THUMBSDOWN");
        }
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
