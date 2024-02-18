// WeaponDetectionSystem.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

// Include Libraries.
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <filesystem>

// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;

// Constants.
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors.
Scalar BLACK = Scalar(0, 0, 0);
Scalar GREEN = Scalar(0, 255, 0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0, 0, 255);

static bool loadCameraParams(const string& filename, Size& imageSize, Size& boardSize, Mat& cameraMatrix, Mat& distCoeffs, float& squareSize, double& totalAvgErr) {
    FileStorage fs(filename, FileStorage::READ);

    imageSize.width = fs["image_width"];
    imageSize.height = fs["image_height"];
    boardSize.width = fs["board_width"];
    boardSize.height = fs["board_height"];
    fs["square_size"] >> squareSize;
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;

    totalAvgErr = fs["avg_reprojection_error"];
    return (fs.isOpened());
}


// Draw the predicted bounding box.
void draw_label(Mat& input_image, string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    // Top left corner.
    Point tlc = Point(left, top);
    // Bottom right corner.
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw black rectangle.
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle.
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}


vector<Mat> pre_process(Mat& input_image, Net& net)
{
    // Convert to blob.
    Mat blob;
    blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate.
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    /*cout << "M = " << endl << " " << outputs.size() << endl << endl;
    cout << "M = " << endl << " " << outputs[0].data[4] << endl << endl;*/

    return outputs;
}

void post_process(Mat& input_image, vector<Mat>& outputs, int classes_number, vector<int>& class_ids, vector<float>& confidences, vector<Rect>& boxes)
{
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float* data = (float*)outputs[0].data;

    auto dimensions = outputs[0].size[2];
    auto rows = outputs[0].size[1];

    /*cout << "Dimesnions = " << endl << " " << dimensions << endl << endl;
    cout << "Rows = " << endl << " " << rows << endl << endl;*/

    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float* classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, classes_number, CV_32FC1, classes_scores);

            //cout << "M = " << endl << " " << scores << endl << endl;

            // Perform minMaxLoc and acquire index of best class score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD)
            {
                // Store class ID and confidence in the pre-defined respective vectors.

                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
            }

        }
        // Jump to the next column.
        data += dimensions;
    }
}

float compute_distance(float focalLenght, float knownWidth, float perWidth) {
    return (knownWidth * focalLenght) / perWidth;
}

Mat draw_boxes(Mat& input_image, const vector<string>& class_name, double focal, vector<double> classes_width ,vector<int>& class_ids, vector<float>& confidences, vector<Rect>& boxes) {
    // Perform Non Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[idx];

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;

        // Draw bounding box.
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3 * THICKNESS);

        float distance = compute_distance(focal, classes_width[class_ids[idx]], width);

        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label + " - d: " + to_string(distance);
        // Draw class labels.
        draw_label(input_image, label, left, top);
    }
    return input_image;
}

Mat post_process(Mat& input_image, vector<Mat>& outputs, const vector<string>& class_name)
{
    // Initialize vectors to hold respective outputs while unwrapping detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float* data = (float*)outputs[0].data;

    auto dimensions = outputs[0].size[2];
    auto rows = outputs[0].size[1];

    /*cout << "Dimesnions = " << endl << " " << dimensions << endl << endl;
    cout << "Rows = " << endl << " " << rows << endl << endl;*/

    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float* classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores);

            // Perform minMaxLoc and acquire index of best class score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD)
            {
                // Store class ID and confidence in the pre-defined respective vectors.

                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
            }

        }
        // Jump to the next column.
        data += dimensions;
    }

    // Perform Non Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[idx];

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;

        // Draw bounding box.
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3 * THICKNESS);

        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        // Draw class labels.
        draw_label(input_image, label, left, top);
    }
    return input_image;
}


int main(int argc, char** argv)
{
    Size boardSize, imageSize;
    Mat cameraMatrix, distCoeffs;
    float squareSize;
    double totalAvgErr = 0;

    string model_name_file;
    string classes_name_file;
    string width_name_file;

    vector<string> class_list;
    vector<double> width_list;
    string line;

    VideoCapture cam;

    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    double focal;

    cv::CommandLineParser parser(argc, argv,
        "{c|../out_camera_data.yml|}"
        "{m|../models/net.onnx|}"
        "{n|../weapon_classes.names|}"
        "{w|../width.names|}"); // to load the configuration file

    if (parser.has("m")) {
        model_name_file=parser.get<string>("m");
    }

    if (parser.has("n")) {
        classes_name_file = parser.get<string>("n");
    }

    if (parser.has("w")) {
        width_name_file = parser.get<string>("w");
    }

    if (parser.has("c")) {
        std::string loadFilename = parser.get<string>("c");
        struct stat buffer;
        if (!(stat(loadFilename.c_str(), &buffer) == 0 && loadCameraParams(loadFilename, imageSize, boardSize, cameraMatrix, distCoeffs, squareSize, totalAvgErr))) {
            return fprintf(stderr, "Failed to open camera parameters files\n"), -1;
        }
    }

    focal = cameraMatrix.at<double>(0, 0);

    const char* winName = "Image View";
    namedWindow(winName, 1);

    //ifstream ifs("coco.names");
    ifstream ifsn(classes_name_file);

    while (getline(ifsn, line))
    {
        class_list.push_back(line);
    }

    if (class_list.size() == 0) {
        return fprintf(stderr, "Failed to load camera names\n"), -1;
    }

    ifstream ifsw(width_name_file);

    while (getline(ifsw, line))
    {
        width_list.push_back(stod(line));
    }

    if (width_list.size() == 0) {
        return fprintf(stderr, "Failed to load default widths\n"), -1;
    }

    if (width_list.size() != class_list.size()) {
        return fprintf(stderr, "Names of classes and default width of the objects are not the same\n"), -1;
    }

    // Load model.
    Net net;
    net = readNet(model_name_file);

    cam.open(0);

    if (!cam.isOpened()) {
        cout << "Fail to open camera" << endl;
        return -1;
    }

    //char key = (char)waitKey(0);

    for (int i = 0;; i++)
    {
        int64 start = cv::getTickCount();

        // Load image.
        Mat frame0;
        Mat frame;
        //frame = imread("sample.jpg");
        if (cam.isOpened()) {
            cam >> frame0;
            frame0.copyTo(frame);
        }

        //int64 start_pre = cv::getTickCount();

        vector<Mat> detections;
        detections = pre_process(frame, net);

        /*double end_pre = (cv::getTickCount() - start_pre)/getTickFrequency();

        std::cout << "Preprocess count : " << end_pre << std::endl;*/

       /* int64 start_post = cv::getTickCount();*/

        Mat temp = frame.clone();
        post_process(temp, detections, class_list.size(), class_ids, confidences, boxes);
        Mat img = draw_boxes(temp, class_list, focal, width_list, class_ids, confidences, boxes);

        /*double end_post = (cv::getTickCount() - start_post) / getTickFrequency();

        std::cout << "Postprocess count : " << end_post << std::endl;*/

        // Put efficiency information.
        // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)

        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time : %.2f ms", t);
        putText(img, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);

        class_ids.clear();
        confidences.clear();
        boxes.clear();

        double fps = cv::getTickFrequency() / (cv::getTickCount() - start);

        string label_fps = "fps: " + to_string(fps);

        int baseline = 0;

        Size textSize = getTextSize(label_fps, FONT_FACE, FONT_SCALE, THICKNESS, &baseline);

        Point textOrg(0,textSize.height);

        putText(img, label_fps, textOrg, FONT_FACE, FONT_SCALE, GREEN);

        imshow(winName, img);

        char c = (char)waitKey(1);
        if (c == 27 || c == 'q' || c == 'Q')
            break;
    }

    return 0;
}
