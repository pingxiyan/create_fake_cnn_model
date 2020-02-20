#include <iostream>

#include <caffe/caffe.hpp>
using namespace caffe;

#include <opencv2/opencv.hpp>


inline void print_fea(std::vector<float>& fea) {
	for(size_t i = 0; i < fea.size(); i++) {
		if(i%10 == 0) {
			std::cout << std::endl;
		}
		std::cout << fea[i] << ", ";
	}
	std::cout << std::endl;
}

void caffe_infer(std::string model_weight, std::string model_prototxt) {
	std::string _dev = "CPU";
	std::string _net_path = model_prototxt;
	std::string _model_path = model_weight;

	bool use_gpu = _dev == "GPU" ? true : false;
	Caffe::set_mode(use_gpu ? Caffe::GPU : Caffe::CPU);
	
	// std::shared_ptr<Net<float> > _net;
	caffe::Net<float>* pvcaffe = new Net<float>(_net_path, caffe::TEST);
	if(pvcaffe == nullptr) {
		std::cout << "New caffe Net fail" << std::endl;
		return;
	}

	/* Load the network. */
	// _net.reset(new Net<float>(net_path, TEST));
	pvcaffe->CopyTrainedLayersFrom(_model_path);

	CHECK_EQ(pvcaffe->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(pvcaffe->num_outputs(), 1) << "Network should have exactly one output.";

	caffe::Blob<float>* input_layer = pvcaffe->input_blobs()[0];
	int _inputChannel = input_layer->channels();
	CHECK(_inputChannel == 3 || _inputChannel == 1) << "Input layer should have 1 or 3 channels.";
	cv::Size _inputSize = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	std::string _feaLayerName = std::string();
	if (_feaLayerName.empty()) {
		size_t last_layer_id = pvcaffe->layer_names().size() - 1;
		_feaLayerName = pvcaffe->layer_names()[last_layer_id];
	}

	// // ************************************
	// CHECK(reinterpret_cast<float*>(_vecInputMat.at(0).data)
	// 		== pvcaffe->input_blobs()[0]->cpu_data()) << "Input channels are not wrapping the input layer of the network.";
	input_layer->Reshape(1, _inputChannel, _inputSize.height, _inputSize.width);
	/* Forward dimension change to all layers. */
	pvcaffe->Reshape();

	auto t1 = std::chrono::high_resolution_clock::now();
	pvcaffe->Forward();
	auto t2 = std::chrono::high_resolution_clock::now();
	printf("one inference time = %f", 
		std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t2-t1).count());

	/* Copy the output layer to a std::vector */
	const boost::shared_ptr<caffe::Blob<float> > feature_blob = pvcaffe->blob_by_name(_feaLayerName);
	float num_imgs = feature_blob->num()/* * total_iter*/;
	float feat_dim = feature_blob->count() / feature_blob->num();

	/* Copy the output layer to a std::vector */
	const float* begin = feature_blob->cpu_data();
	const float* end = begin + feature_blob->channels();
	std::vector<float> features = std::vector<float>(begin, end);
	
	print_fea(features);
}

int main(int argc, char** argv) {
	std::string model_weight = "../../fake_tiny_yolov2/fake_model.caffemodel";
	std::string model_prototxt = "../../fake_tiny_yolov2/fake_model.prototxt";
	std::string image_fn = "";

	if(argc == 2 && argv[1] == std::string("-h")) {
		std::cout << "$ ./caffe_test -h" << std::endl;
		std::cout << "argv[1]: caffe.prototxt" << std::endl;
		std::cout << "argv[2]: caffe.model" << std::endl;
		return EXIT_SUCCESS;
	}
	if(argc == 3) {
		model_prototxt = std::string(argv[1]);
		model_weight = std::string(argv[2]);
	}
	
	std::cout << "model_weight = " << model_weight << std::endl;
	std::cout << "model_prototxt = " << model_prototxt << std::endl;

	caffe_infer(model_weight, model_prototxt);

	return 0;
}
