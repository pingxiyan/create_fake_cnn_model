//
// Copyright 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//


#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <tuple>
#include <list>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include "region_yolov2tiny.h"

#include "zhangcong_detected_object.h"
#include "zhangcong_tinyyolo2_post.h"

using namespace InferenceEngine;

Blob::Ptr yoloLayer_yolov2tiny(const Blob::Ptr &lastBlob, int inputHeight, int inputWidth) {
    const TensorDesc quantTensor = lastBlob->getTensorDesc();
    const TensorDesc outTensor = TensorDesc(InferenceEngine::Precision::FP32,
        {1, 1, 13*13*5*20, 7},
        lastBlob->getTensorDesc().getLayout());

    Blob::Ptr outputBlob = make_shared_blob<float>(outTensor);
    outputBlob->allocate();

    const float *inputRawData = lastBlob->cbuffer().as<const float *>();
    float *outputRawData = outputBlob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();

    int shape[]={13, 13, 5, 25};
    int strides[]={13*128, 128, 25, 1};
    postprocess::yolov2(inputRawData, shape, strides,
        0.4f, 0.45f, 20, 416, 416, outputRawData);

    return outputBlob;
}

void putImg2InputBlob(const cv::Mat& img, Blob::Ptr inputBlob)
{
	uint8_t * blob_data = inputBlob->buffer().as<uint8_t*>();

    cv::Mat rsz;
    cv::resize(img, rsz, cv::Size(416,416));

	for (int c = 0; c < 3; c++) {
		for (int h = 0; h < rsz.rows; h++) {
			for (int w = 0; w < rsz.cols; w++) {
				blob_data[c * rsz.rows * rsz.cols + h * rsz.cols + w] = 
                    rsz.at<cv::Vec3b>(h, w)[c];
			}
		}
	}
}

/**
* @brief The entry point the Inference Engine sample application
* @file detection_sample/main.cpp
* @example detection_sample/main.cpp
*/
int main(int argc, char *argv[]) {
    std::string modelXml = "../../fake_model.xml";
    std::string modelBin = "../../fake_model.bin";
    modelXml = "../darknet_tiny_yolo_voc/optimized/tiny_yolo_v2.xml";
    modelBin = "../darknet_tiny_yolo_voc/optimized/tiny_yolo_v2.bin";
    std::string imgFN = "../fish-bike.jpg";
    imgFN = "../../caffe/examples/images/cat.jpg";
    imgFN = "../323934632.jpg";
    imgFN = "../503911236.jpg";
    imgFN = "../../test_img2.jpg";
    std::string dev = "CPU";
    if(argc == 2 && argv[1] == std::string("-h")) {
        std::cout << "argv[1] = xml" << std::endl;
        std::cout << "argv[2] = img()" << std::endl;
        std::cout << "argv[3] = CPU/GPU/HDDL" << std::endl;
        return EXIT_SUCCESS;
    }
    else if(argc == 5){
        modelXml = argv[1];
        modelBin = argv[2];
        imgFN = argv[3];
        dev = argv[4];
    }

    std::cout << "modelXml = " << modelXml << std::endl;
    std::cout << "modelBin = " << modelBin << std::endl;
    std::cout << "imgFN = " << imgFN << std::endl;
    std::cout << "dev = " << dev << std::endl;

    cv::Mat inputMat = cv::imread(imgFN);
    if(inputMat.empty()) {
        std::cout << "Can't imread: " << imgFN << std::endl;
        return EXIT_FAILURE;
    }

    try {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;
        
        std::cout << "Creating Inference Engine" << std::endl;
        
        Core ie;
        // std::cout << ie.GetVersions(dev) << std::endl;

        /*If CPU device, load default library with extensions that comes with the product*/
        // if (std::string("CPU") == dev) {
        //     /**
        //     * cpu_extensions library is compiled from "extension" folder containing
        //     * custom MKLDNNPlugin layer implementations. These layers are not supported
        //     * by mkldnn, but they can be useful for inferring custom topologies.
        //     **/
        //     ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
        // }

        std::string binFileName = modelBin;

        CNNNetReader networkReader;
        /** Read network model **/
        networkReader.ReadNetwork(modelXml);

        /** Extract model name and load weights **/
        networkReader.ReadWeights(binFileName);
        CNNNetwork network = networkReader.getNetwork();

        InputsDataMap inputsInfo(network.getInputsInfo());
        if (inputsInfo.size() != 1 && inputsInfo.size() != 2) throw std::logic_error("Sample supports topologies only with 1 or 2 inputs");

        auto inputInfoItem = *inputsInfo.begin();

        /** Specifying the precision and layout of input data provided by the user.
         * This should be called before load of the network to the device **/
        inputInfoItem.second->setPrecision(Precision::U8);
        inputInfoItem.second->setLayout(Layout::NCHW);

        ExecutableNetwork executable_network = ie.LoadNetwork(network, dev);
        InferRequest infer_request = executable_network.CreateInferRequest();

        // --------------------------- 5. Prepare input --------------------------------------------------------
        /** Creating input blob **/
        Blob::Ptr inputBlob = infer_request.GetBlob(inputsInfo.begin()->first);
        if (!inputBlob) {
            throw std::logic_error("Cannot get input blob from inferRequest");
        }
        putImg2InputBlob(inputMat, inputBlob);

        infer_request.Infer();
        std::cout << "infer_request completed successfully" << std::endl;
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Process output -------------------------------------------------------
        std::cout << "Processing output blobs" << std::endl;
        OutputsDataMap outputInfo(network.getOutputsInfo());
        if (outputInfo.size() != 1) throw std::logic_error("Sample supports topologies with 1 output only");
        Blob::Ptr outputBlob = infer_request.GetBlob(outputInfo.begin()->first);

        std::vector<DetectedObject_t> rlst = YoloV2Tiny::TensorToBBoxYoloV2TinyCommon(outputBlob, 
            inputMat.rows, inputMat.cols, 0.4, YoloV2Tiny::fillRawNetOut);

        for(auto r : rlst) {
            std::cout << r.x << ", " << r.y << ", "  << r.width << ", "  <<
                r.height << ", "  << r.confidence << ", " << std::endl;
            
            cv::Rect rt = cv::Rect(r.x, r.y, r.width, r.height);
            cv::rectangle(inputMat, rt, cv::Scalar(0,0,255), 1);

        }
        cv::imshow("s", inputMat);
        cv::waitKey(0);

        // std::cout << "yoloLayer_yolov2tiny" << std::endl;
        // // Region YOLO layer
        // Blob::Ptr detectResult = yoloLayer_yolov2tiny(outputBlob, inputMat.rows, inputMat.cols);

        // // Print result.
        // std::cout << "Print result" << std::endl;
        // size_t N = detectResult->getTensorDesc().getDims()[2];
        // if (detectResult->getTensorDesc().getDims()[3] != 7) {
        //     throw std::logic_error("Output item should have 7 as a last dimension");
        // }
        // const float *rawData = detectResult->cbuffer().as<const float *>();
        // imageid,labelid,confidence,x0,y0,x1,y1
        // for (size_t i = 0; i < N; i++) {
        //     if (rawData[i*7 + 2] > 0.001) {
        //         std::cout << "confidence = " << rawData[i*7 + 2] << std::endl;
        //         std::cout << "x0,y0,x1,y1 = " << rawData[i*7 + 3] << ", "
        //             << rawData[i*7 + 4] << ", "
        //             << rawData[i*7 + 5] << ", "
        //             << rawData[i*7 + 6] << std::endl;
        //     }
        // }

    }
    catch (const std::exception& error) {
        std::cout << "" << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cout << "Unknown/internal exception happened." << std::endl;
        return 1;
    }

    std::cout << "Execution successful" << std::endl;
    return 0;
}
