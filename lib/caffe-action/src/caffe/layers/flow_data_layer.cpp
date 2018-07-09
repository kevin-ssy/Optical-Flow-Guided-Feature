////
//// Created by kevin on 2/14/17.
////
//
//
//
//#include <stdint.h>
//
//#include <vector>
//#include <string>
//#include "caffe/common.hpp"
//#include "caffe/data_layers.hpp"
//#include "caffe/layer.hpp"
//#include "caffe/proto/caffe.pb.h"
//#include "caffe/util/benchmark.hpp"
//#include "caffe/util/io.hpp"
//#include "caffe/util/math_functions.hpp"
//#include "caffe/util/rng.hpp"
//#include <opencv2/core/core.hpp>
//#include <opencv2/opencv.hpp>
//#include "boost/algorithm/string/split.hpp"
//#include "boost/algorithm/string/classification.hpp"
//
//
//#define ADD_IMGPREFIX(x, y) (x+y)
//using namespace cv;
//// using namespace std;
//namespace caffe {
//
//	template<typename Dtype>
//	FlowDataLayer<Dtype>::FlowDataLayer(const LayerParameter &param)
//			: BasePrefetchingDataLayer<Dtype>(param) {
//	}
//
//	template<typename Dtype>
//	FlowDataLayer<Dtype>::~FlowDataLayer() {
//		this->JoinPrefetchThread();
//	}
//
//	template<typename Dtype>
//	void FlowDataLayer<Dtype>::read_file(const std::string &file_name, std::string &img_prefix, const int batch_size) {
//		string line;
////		stringstream buffer;
////		string file_path = prefix + "/" + file_name;o
//
//		std::ifstream myfile(file_name.c_str());
//		if (myfile.is_open()) {
//			while (getline(myfile, line)) {
//				vector<string> line_vector;
//				boost::split(line_vector, line, boost::is_any_of(" "));
//				if (line_vector[0].compare(line_vector[1]) == 0) continue;
//				this->total_image_info.push_back(new ImagePair(ADD_IMGPREFIX(img_prefix, line_vector[0]),
//															   ADD_IMGPREFIX(img_prefix, line_vector[1]),
//															   ADD_IMGPREFIX(img_prefix, line_vector[2]),
//															   ADD_IMGPREFIX(img_prefix, line_vector[3])));
//
//			}
//			myfile.close();
//		} else
//			LOG(INFO) << "Unable to open file:" << file_name;
//
//		LOG(INFO) << "We have " << this->total_image_info.size() << " pairs in total.";
//		random_shuffle_totalmetadata(batch_size);
//		LOG(INFO) << "Json parsing finished!";
//	}
//
//	template<typename Dtype>
//	void FlowDataLayer<Dtype>::random_shuffle_totalmetadata(const int batch_size) {
//		LOG(INFO) << "RESTART AND FETCH IMAGE FROM BEGINING";
//		LOG(INFO) << "SHUFFLING IMAGE...";
//		this->item_id = 0;
//		for (int i = 0; i < this->total_image_info.size() / 2; i++) {
//			unsigned long swap_index =(unsigned long) rand() % (this->total_image_info.size() / 2) +
//									  (this->total_image_info.size() / 2 - 1);
//			ImagePair meta_swap = this->total_image_info[swap_index];
//			this->total_image_info[swap_index] = this->total_image_info[i];
//			this->total_image_info[i] = meta_swap;
//
//		}
//	}
//
//
//	template<typename Dtype>
//	void FlowDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
//											   const vector<Blob<Dtype> *> &top) {
//		// In this part, we just initialize the blob in data layer.
//		// The top shape should be determined by the shape of image,
//		// thus an image is required to be loaded for initialization.
//		// Besides, the image path is stored in json, thus the metadata
//		// from json is initialized here. We DO NOT need to read json
//		// elsewhere.
//		// the database and img_prefix should be written in pair
////
//
//		int num_of_database = this->layer_param_.video_data_param().json_path().size();
//		int num_of_prefix = this->layer_param_.video_data_param().img_prefix().size();
//		const int crop_size = this->layer_param_.transform_param().crop_size();
//
//		const bool is_coor = this->layer_param_.video_data_param().is_coor();
//		const int batch_size = this->layer_param_.video_data_param().batch_size();
//		item_id = 0;
//		CHECK_EQ(num_of_database, num_of_prefix);
//
//		// read database in group
//		for (int database_id = 0; database_id < num_of_database; database_id++) {
//
//			std::string json_path_str = this->layer_param_.video_data_param().json_path(database_id);
//			std::string img_prefix_str = this->layer_param_.video_data_param().img_prefix(database_id);
////			LOG(INFO) << "2";
//			read_file(json_path_str, img_prefix_str, batch_size);
//		}
//// start reading image
////		LOG(INFO) << "3";
////		LOG(INFO) << "out size: " << this->total_image_info.size();
//		ImagePair image_info = this->total_image_info.front();
////		LOG(INFO) << "4";
//		cv::Mat img1 = cv::imread(image_info.img1);
//		cv::Mat img2 = cv::imread(image_info.img2);
//		cv::Mat flow1 = cv::imread(image_info.flow1);
//		cv::Mat flow2 = cv::imread(image_info.flow2);
////		LOG(INFO) << "5";
//		if (img1.empty() || img2.empty() || flow1.empty() || flow2.empty()) {
//			LOG(ERROR) << "Cannot open image:" << image_info.img1;
//			LOG(ERROR) << "Cannot open image:" << image_info.img2;
//			LOG(ERROR) << "Cannot open image:" << image_info.flow1;
//			LOG(ERROR) << "Cannot open image:" << image_info.flow2;
//		}
//
//		// Read a data point, and use it to initialize the top blob.
//		CHECK_EQ(img1.rows, img2.rows);
//		CHECK_EQ(img1.cols, img2.cols);
//		CHECK_EQ(img1.rows, flow1.rows);
//		CHECK_EQ(img1.cols, flow1.cols);
//		CHECK_EQ(img1.rows, flow2.rows);
//		CHECK_EQ(img1.cols, flow2.cols);
//		LOG(INFO) << img1.rows << " " << img1.cols << " " << img1.channels();
//
//		// image
//
//		if (is_coor){ // if use the FlownetCoor architecture, each data layer just read only one image
//			if (crop_size > 0) {
//				top[0]->Reshape(batch_size, img1.channels(), crop_size, crop_size);
//				this->prefetch_data_.Reshape(batch_size, img1.channels(), crop_size, crop_size);
//				this->transformed_data_.Reshape(1, img1.channels(), crop_size, crop_size);
//			} else {
//				const int height = this->layer_param_.transform_param().crop_size_y();
//				const int width = this->layer_param_.transform_param().crop_size_x();
//				top[0]->Reshape(batch_size, 2 * img1.channels(), height, width);
//				this->prefetch_data_.Reshape(batch_size, 2 * img1.channels(), height, width);
//				this->transformed_data_.Reshape(1, 3, height, width);
//			}
//		} else {
//			if (crop_size > 0) {
//				top[0]->Reshape(batch_size, 2 * img1.channels(), crop_size, crop_size);
//				this->prefetch_data_.Reshape(batch_size, 2 * img1.channels(), crop_size, crop_size);
//				this->transformed_data_.Reshape(1, 2 * img1.channels(), crop_size, crop_size);
//			} else {
//				const int height = this->layer_param_.transform_param().crop_size_y();
//				const int width = this->layer_param_.transform_param().crop_size_x();
//				top[0]->Reshape(batch_size, 2 * img1.channels(), height, width);
//				this->prefetch_data_.Reshape(batch_size, 2 * img1.channels(), height, width);
//				this->transformed_data_.Reshape(1, 6, height, width);
//			}
//		}
//
//		LOG(INFO) << "output data size: " << top[0]->num() << ","
//				  << top[0]->channels() << "," << top[0]->height() << ","
//				  << top[0]->width();
//
//		// label
//		if (this->output_labels_) {
//			const int stride = this->layer_param_.transform_param().stride();
//			const int height = this->layer_param_.transform_param().crop_size_y();
//			const int width = this->layer_param_.transform_param().crop_size_x();
//
////			int num_parts = this->layer_param_.transform_param().num_parts();
//			top[1]->Reshape(batch_size, 2, height, width);
//			this->prefetch_label_.Reshape(batch_size, 2, height, width);
//			this->transformed_label_.Reshape(1, 2, height, width);
//		}
//
//	}
//
//// This function is called on prefetch thread
//	template<typename Dtype>
//	void FlowDataLayer<Dtype>::InternalThreadEntry() {
//		CHECK(this->transformed_data_.count());
////		const int thread_num = Caffe::getThreadNum();
////		const int thread_id = Caffe::getThreadId();
//		const int batch_size = this->layer_param_.video_data_param().batch_size() / thread_num;
//		if (item_id >= total_image_info.size()) {
//			LOG(INFO) << "UNEXPECTEDLY ENCOUNTERED SHUFFLE!!!";
//			LOG(INFO) << "CURRENT item_id: " << item_id;
//			random_shuffle_totalmetadata(batch_size);
//		}
//		CPUTimer batch_timer;
//
//		ImagePair *image_pair = &this->total_image_info[item_id];
//		cv::Mat img1 = cv::imread(image_pair->img1);
//		cv::Mat img2 = cv::imread(image_pair->img2);
//		cv::Mat flow1 = cv::imread(image_pair->flow1);
//		cv::Mat flow2 = cv::imread(image_pair->flow2);
//
//		batch_timer.Start();
//		double trans_time = 0;
//		static int cnt = 0;
//		CPUTimer timer;
//
//
//		// Reshape on single input batches for inputs of varying dimension.
//
//
//		Dtype *top_data = this->prefetch_data_.mutable_cpu_data();
//		Dtype *top_label = this->prefetch_label_.mutable_cpu_data();  // suppress warnings about uninitialized variables
//		for (int i = 0; i < batch_size; ++i) {
//			// Apply data transformations (mirror, scale, crop...)
//			timer.Start();
//			if (item_id < total_image_info.size()) {
//
//				const int offset_data = this->prefetch_data_.offset(i);
//				const int offset_label = this->prefetch_label_.offset(i);
//				this->transformed_data_.set_cpu_data(top_data + offset_data);
//				this->transformed_label_.set_cpu_data(top_label + offset_label);
//
//				this->data_transformer_->Transform_flow(img1, img2, flow1, flow2, &(this->transformed_data_),
//														&(this->transformed_label_), cnt);
//
//				++cnt;
//				trans_time += timer.MicroSeconds();
//
//				image_pair = &this->total_image_info[item_id];
//				img1 = cv::imread(image_pair->img1);
//				img2 = cv::imread(image_pair->img2);
//				flow1 = cv::imread(image_pair->flow1);
//				flow2 = cv::imread(image_pair->flow2);
//
////                    LOG(INFO) << "item_id: " << item_id<< ", image path: " << img_path;
//				item_id++;
//			} else {
//				LOG(INFO) << "RESTART AND FETCH IMAGE FROM BEGINING";
//				LOG(INFO) << "SHUFFLING IMAGE...";
//				random_shuffle_totalmetadata(batch_size);
//			}
////                imshow(img_path, img);
////                waitKey(0);
////            LOG(INFO) << "NEW IMG_PATH: " << img_path;
//		}
////		item_id + batch_size - 1 >= total_image_info.size() ?
////				item_id += batch_size * (thread_num - 1) : item_id += batch_size * thread_num;
//		batch_timer.Stop();
//#ifdef BENCHMARK_DATA
//		LOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
//			LOG(INFO) << "  Dequeue time: " << deque_time / 1000 << " ms.";
//			LOG(INFO) << "   Decode time: " << decod_time / 1000 << " ms.";
//			LOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
//#endif
//	}
//
//	INSTANTIATE_CLASS(FlowDataLayer);
//
//	REGISTER_LAYER_CLASS(FlowData);
//
//}  // namespace caffe
