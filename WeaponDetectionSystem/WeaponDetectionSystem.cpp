// Librerie.
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

// Constanti.
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.4;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;
const int PIXEL_FOR_BB = 10;
const int DANGER_DISTANCE = 100;

// Parametri del testo della detection.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Quanti frame devo aspettare senza detection prima di chiudere il video
const int FRAME_SKIP_VIDEO = 30;
const string PATH_FOR_VIDEO= "detectionVideo";

vector<Mat> video_frames;
int frame_counter = 0;

// Colori.
Scalar BLACK = Scalar(0, 0, 0);
Scalar GREEN = Scalar(0, 255, 0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0, 0, 255);

// Dato il nome del file contenente la calibrazione della camera lo apre e ne estrae alcuni parametri.
static bool loadCameraParams(const string& filename, Mat& cameraMatrix, Mat& distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return (fs.isOpened());
}

// Disegna l'etichetta di una bounding box.
void draw_label(Mat& input_image, string label, int left, int top)
{
    int baseLine; // variabile per inserire l'etichetta in cima alla bounding box
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top - label_size.height - PIXEL_FOR_BB, 0); // 0 per rimanere nella schermata in caso vada fuori

    // angoli della bounding box
    Point topLeftCorner = Point(left, top);
    Point bottomRightCorner = Point(left + label_size.width, top + label_size.height + baseLine);

    // disegna un rettangolo nero ed inserisce l'etichetta al suo interno
    rectangle(input_image, topLeftCorner, bottomRightCorner, BLACK, FILLED);
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

// Viene impostata l'immagine come input della rete per effettuare il passo di inferenza ed ottenere i risultati.
vector<Mat> pre_process(Mat& input_image, Net& net)
{
    // convezione dell'immagine in blob.
    Mat blob;
    blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    // inferenza a partire dall'immagine di input per ottenere la classificazione degli oggetti e le relative bounding box
    net.setInput(blob);
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    return outputs;
}

// Estrae le informazioni dall'output della rete, cioè le coordinate delle bounding box e la classe dell'oggetto identificato e le restituisce negli ultimi 3 parametri
void post_process(Mat& input_image, vector<Mat>& outputs, int classes_number, vector<int>& class_ids, vector<float>& confidences, vector<Rect>& boxes)
{
    // il fattore di scala è necessario perché YOLO utilizza un formato specifico
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    // l'output è una matrice di 1x25200x{numero di classi}, quindi viene preso direttamente il primo elemento
    float* data = (float*)outputs[0].data;

    auto dimensions = outputs[0].size[2]; // il numero delle classi che la rete riconosce
    auto rows = outputs[0].size[1]; // numero di detection effettuate dalla rete

    // vengono iterate le 25200 possibili detections che vengono effettuate per ogni immagine.
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // le detections con confidence bassa vengono scartate.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            // puntatore alle confidence delle singole classi
            float* classes_scores = data + 5; // 5 perché la confidence è dopo le celle di x e y del centro, della lungezza e dell'altezza
            // matrice contenente tutti gli score delle singole classi
            Mat scores(1, classes_number, CV_32FC1, classes_scores);

            // viene effettuata la minMaxLoc per ottenere solo la classe con il miglior risultato per la detection di quest'oggetto
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            // se una delle classi individuate ha una confidence accettabile, la sua bounding box verrà disegnata
            if (max_class_score > SCORE_THRESHOLD)
            {
                // confidence e tipo di classe vengono inseriti nei vettori utilizzati per restituire i risultati
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // coordinate del centro della bounding box e dimensioni
                float cx = data[0];
                float cy = data[1];
                float w = data[2];
                float h = data[3];

                // vengono calcolate le coordinate dei vertici per disegnare la bounding box ed inserite nel vettore dei risultati
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
        // il puntatore viene fatto arrivare alla prossima detection della matrice
        data += dimensions;
    }
}

// Calcola la distanza a partire dalla lunghezza focale, dalla lunghezza dell'oggetto e dal numero di pixel che occupa in lunghezza.
float compute_distance(float focalLenght, float knownWidth, float perWidth) {
    return (knownWidth * focalLenght) / perWidth;
}

void video_manager(Mat& nextFrame, bool save) {

    if (save) {
        video_frames.push_back(nextFrame);
        frame_counter = 0;
    }
    else {
        frame_counter++;
    }

    if (frame_counter >= FRAME_SKIP_VIDEO && video_frames.size() > 0 && !save) {

        std::time_t t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
        string time_stamp = oss.str();

        string file_name = PATH_FOR_VIDEO + "\\" + time_stamp + "_detection.avi";

        VideoWriter video = VideoWriter(file_name, VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(nextFrame.size[1], nextFrame.size[0]));

        for (int i = 0; i < video_frames.size(); i++) {
            video.write(video_frames[i]);
        }

        video.release();
        video_frames.clear();
        frame_counter = 0;
    }

}

// Disegna le bounding box degli oggetti rilevati eliminando quelle sovrapposte
Mat draw_boxes(Mat& input_image, const vector<string>& class_name, double focal, vector<double> classes_width ,vector<int>& class_ids, vector<float>& confidences, vector<Rect>& boxes) {
    // rimuove i rilevamenti sovrapposti con la Non Maximum Suppression utilizzando il rapporto tra l'intersezione e l'unione delle box
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    bool save_frame_video = false;

    for (int i = 0; i < indices.size(); i++)
    {
        // prende i riferimenti della bounding box rilevata
        int idx = indices[i];
        Rect box = boxes[idx];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;

        // disegna bounding box
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3 * THICKNESS);

        // calcola la distanza e comunica il grado di minaccia
        float distance = compute_distance(focal, classes_width[class_ids[idx]], max(width, height));
        if (distance < DANGER_DISTANCE) {
            draw_label(input_image, "DANGER", left, top + height);
            save_frame_video = true;
        }
        else {
            draw_label(input_image, "ATTENTION", left, top + height);
        }

        // disegna l'etichetta con il nome della classe e la confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label + " - d: " + to_string(distance);
        draw_label(input_image, label, left, top);
    }

    video_manager(input_image, save_frame_video);

    return input_image;
}

static void help(char** argv)
{
    printf("This is an application for the detection of the weapons.\n"
        "Usage: %s\n\n"
        "     [-c=<camera_params>]    # the filename for intrinsic [and extrinsic] parameters of the camera\n"
        "     [-m=<net>]              # the filename of net to use for the detection\n"
        "     [-n=<classes_names>]    # the filename with the names of the classes detected by the net\n"
        "     [-w=<classes_width>]    # the filename with the width of the objects detected by the net\n"
        "     [-l]                    # print some informations about the performance\n"
        "\n"
        "Example command line to use the program:\n"
        "   ./x64/Release/WeaponDetectionSystem.exe -c=camera_data.yml -m=models/net.onnx -n=weapon_classes.names -w=width.names -l"
        "\n\n"
        "To quit the program use the key: <ESC>, 'q' or 'Q'\n", argv[0]
    );
}

// Main del programma.
int main(int argc, char** argv)
{
    Mat cameraMatrix, distCoeffs;

    string model_name_file;
    string classes_name_file;
    string width_name_file;
    bool log = false;

    vector<string> class_list;
    vector<double> width_list;
    string line;

    VideoCapture cam;

    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    double focal;

    // il parser predispone l'utilizzo degli argomenti ricevuti e definisce i valori di default
    CommandLineParser parser(argc, argv,
        "{help||}"
        "{c|../camera_data.yml|}"
        "{m|../models/net.onnx|}"
        "{n|../weapon_classes.names|}"
        "{w|../width.names|}"
        "{l||}");
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
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
        if (!(stat(loadFilename.c_str(), &buffer) == 0 && loadCameraParams(loadFilename, cameraMatrix, distCoeffs))) {
            return fprintf(stderr, "Failed to open camera parameters files\n"), -1;
        }
    }
    if (parser.has("l")) {
        log = true;
    }

    // la distanza focale viene calcolata come media tra quella orizzontale e quella verticale
    focal = (cameraMatrix.at<double>(0, 0) + cameraMatrix.at<double>(1, 1)) / 2;

    // lettura del file con i nomi delle classi
    ifstream ifsn(classes_name_file);
    while (getline(ifsn, line))
    {
        class_list.push_back(line);
    }
    if (class_list.size() == 0) {
        return fprintf(stderr, "Failed to load camera names\n"), -1;
    }

    // lettura del file con le lunghezze medie degli oggetti delle classi
    ifstream ifsw(width_name_file);
    while (getline(ifsw, line))
    {
        width_list.push_back(stod(line));
    }
    if (width_list.size() == 0) {
        return fprintf(stderr, "Failed to load default widths\n"), -1;
    }

    // se il numero delle classi non coincide viene comunicato un errore
    if (width_list.size() != class_list.size()) {
        return fprintf(stderr, "Names of classes and default width of the objects are not the same\n"), -1;
    }

    // caricamento del file contenente il modello addestrato per il riconoscimento
    Net net;
    net = readNet(model_name_file);

    // creazione della GUI e apertura delle camera
    const char* winName = "Image View";
    namedWindow(winName, WINDOW_AUTOSIZE);
    cam.open(0);
    if (!cam.isOpened()) {
        cout << "Fail to open camera" << endl;
        return -1;
    }

    namespace fs = std::filesystem;
    fs::create_directory(PATH_FOR_VIDEO);

    // ciclo infinito
    for (int i = 0;; i++)
    {
        // viene utilizzato per il calcolo degli FPS
        int64 start = getTickCount();

        // viene catturata l'immagine e copiata per utilizzarla in sicurezza
        Mat frame0;
        Mat frame;
        if (cam.isOpened()) {
            cam >> frame0;
            frame0.copyTo(frame);
        }

        // se si vuole il log delle performance viene calcolato il tempo necessario al preprocessing
        int64 start_pre;
        if (log) {
            start_pre = getTickCount();
        }

        // viene corretta la distorsione dell'immagine dovuta alla lente della camera
        Mat distorted = frame.clone();
        undistort(distorted, frame, cameraMatrix, distCoeffs);

        // viene eseguito il preprocessing per ottenere le detecions
        vector<Mat> detections;
        detections = pre_process(frame, net);

        // se si vuole il log delle performance viene stampato il tempo necessario al preprocessing e calcolato quello per il posprocessing
        int64 start_post;
        if (log) {
            double end_pre = (getTickCount() - start_pre) / getTickFrequency();
            std::cout << "Preprocess count : " << end_pre << std::endl;
            start_post = getTickCount();
        }

        // viene effettuato il post processing per recuperare le bounding box degli oggetti rilevati che vengono poi disegnati
        Mat temp = frame.clone();
        post_process(temp, detections, class_list.size(), class_ids, confidences, boxes);
        Mat img = draw_boxes(temp, class_list, focal, width_list, class_ids, confidences, boxes);

        // vengono pulite le variabili per utilizzarle nella prossima iterazione
        class_ids.clear();
        confidences.clear();
        boxes.clear();

        // se si vuole il log delle performance viene stampato il tempo necessario al postprocessing
        if (log) {
            double end_post = (getTickCount() - start_post) / getTickFrequency();
            std::cout << "Postprocess count : " << end_post << std::endl;
        }

        // viene calcolato il tempo totale impiegato per effettuare l'inferenza
        if (log) {
            vector<double> layersTimes;
            double freq = getTickFrequency() / 1000; // da secondi a millisecondi
            double t = net.getPerfProfile(layersTimes) / freq;
            string label = format("Inference time : %.2f ms", t);
            putText(img, label, Point(0, 40), FONT_FACE, FONT_SCALE, RED);
        }

        // vengono calcolati e mostrati gli FPS
        double fps = getTickFrequency() / (getTickCount() - start);
        string label_fps = "FPS: " + to_string(fps);
        int baseline = 0;
        Size textSize = getTextSize(label_fps, FONT_FACE, FONT_SCALE, THICKNESS, &baseline);
        Point textOrg(0,textSize.height);
        putText(img, label_fps, textOrg, FONT_FACE, FONT_SCALE, GREEN);

        // viene mostrata l'immagine contenente le bounding box e gli altri testi
        imshow(winName, img);

        // se l'utente clicca "ESC", "q" o "Q" il programma termina
        char c = (char)waitKey(1);
        if (c == 27 || c == 'q' || c == 'Q')
            break;
    }

    return 0;
}
