//
// Created by kevin on 7/14/17.
//
#include "caffe/layers/edge_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/filler.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "boost/smart_ptr/make_shared.hpp"


#ifdef USE_CUDNN
#include "caffe/vision_layers.hpp"
#endif


namespace caffe {


	template<typename Dtype>
	void visualize_edge(const Dtype *data, int width, int height, string img_key) {
		cv::Mat img1 = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
		double min = 0;
		double max = 0;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(img1, &min, &max);
		double range = max - min;
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++) {
				unsigned char &rgb = img1.at<uchar>(y, x);
				rgb = static_cast<unsigned char>(data[y * width + x] > 0 ? data[y * width + x] : -(data[y * width + x]));
			}

//		cv::Mat output;
//		img1.convertTo(output, CV_8U, 255 / range);
//		cv::normalize(img1, output, 0, 255, cv::NORM_MINMAX, CV_8UC1);
//		cv::convertScaleAbs(img1, output);
		cv::imwrite("visualize/" + img_key + "_edge.jpg", img1);
		LOG(INFO) << "Img:" << img_key << " wrote.";
	}



	template<typename Dtype>
	void reset_kernel_x(Dtype* weights,const int num_kernels) {
		for (int n = 0; n < num_kernels; n++) {
			int i = n * 9;  // 3 x 3 filter
//			weights[i + 0] = -1;
//			weights[i + 1] =  0;
//			weights[i + 2] =  1;
//			weights[i + 3] = -2;
//			weights[i + 4] =  0;
//			weights[i + 5] =  2;
//			weights[i + 6] = -1;
//			weights[i + 7] =  0;
//			weights[i + 8] =  1;
			weights[i + 0] = 0;
			weights[i + 1] = 0;
			weights[i + 2] = 0;
			weights[i + 3] = -1;
			weights[i + 4] = 0;
			weights[i + 5] = 1;
			weights[i + 6] = 0;
			weights[i + 7] = 0;
			weights[i + 8] = 0;
		}
	}

	template<typename Dtype>
	void reset_kernel_y(Dtype* weights, const int num_kernels) {
		for (int n = 0; n < num_kernels; n++) {
			int i = n * 9;  // 3 x 3 filter
//			weights[i + 0] =  1;
//			weights[i + 1] =  2;
//			weights[i + 2] =  1;
//			weights[i + 3] =  0;
//			weights[i + 4] =  0;
//			weights[i + 5] =  0;
//			weights[i + 6] = -1;
//			weights[i + 7] = -2;
//			weights[i + 8] = -1;
			weights[i + 0] = 0;
			weights[i + 1] = 1;
			weights[i + 2] = 0;
			weights[i + 3] = 0;
			weights[i + 4] = 0;
			weights[i + 5] = 0;
			weights[i + 6] = 0;
			weights[i + 7] = -1;
			weights[i + 8] = 0;
		}
	}

	template<typename Dtype>
	void reset_kernel_sum(Dtype* weights, const int num_kernels) {
		for (int n = 0; n < num_kernels; n++) {
			int i = n * 9;  // 3 x 3 filter
//			weights[i + 0] =  1;
//			weights[i + 1] =  2;
//			weights[i + 2] =  1;
//			weights[i + 3] =  0;
//			weights[i + 4] =  0;
//			weights[i + 5] =  0;
//			weights[i + 6] = -1;
//			weights[i + 7] = -2;
//			weights[i + 8] = -1;
			weights[i + 0] = 0;
			weights[i + 1] = 1;
			weights[i + 2] = 0;
			weights[i + 3] = -1;
			weights[i + 4] = 0;
			weights[i + 5] = 1;
			weights[i + 6] = 0;
			weights[i + 7] = -1;
			weights[i + 8] = 0;
		}
	}


	template<typename Dtype>
void EdgeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
const vector<Blob<Dtype> *> &top) {

//		shared_ptr<ConstantFiller<Dtype> > filler;
//		FillerParameter filler_param;
//		filler_param.set_value(0.);
//		filler.reset(new ConstantFiller<Dtype>(filler_param));
//		filler->Fill(bottom[0]);

Dtype* weights = this->layer->blobs()[0]->mutable_cpu_data();
	const int num_kernels = bottom[0]->channels();
	const int num_channels = this->layer_param_.edge_param().num_channels() == 0 ? num_kernels :
							 this->layer_param_.edge_param().num_channels();
if (this->layer_param_.edge_param().orient() == EdgeParameter_Orientation_x){
reset_kernel_x(weights, num_kernels);
} else if (this->layer_param_.edge_param().orient() == EdgeParameter_Orientation_y) {
reset_kernel_y(weights, num_kernels);
}
/*TODO:Uncomment this to visualize
int rand_id = rand() % 100000;
for (int n = 0; n < bottom[0]->shape(0); n++) {
for (int c = 0; c < bottom[0]->shape(1); c++){
std::stringstream ss_input;
ss_input << rand_id << "_" << bottom[0]->shape(3) << "_" << bottom[0]->shape(2) << "img_bottom" << n << "_" << c << "_edge";
const int img_pos = n * bottom[0]->shape(3) * bottom[0]->shape(2) * bottom[0]->shape(1);
const int ch_pos = c * bottom[0]->shape(3) * bottom[0]->shape(2);
visualize_edge(bottom[0]->cpu_data() + img_pos + ch_pos, bottom[0]->shape(3), bottom[0]->shape(2), ss_input.str());
LOG(INFO) << "GPU CALL: LABEL SPAPE: " << bottom[0]->shape(0) << " " << bottom[0]->shape(1) << " " <<
		  bottom[0]->shape(2) << " " << bottom[0]->shape(3);
}

}
// */
this->layer->Forward(conv_bottom, conv_top);
//		LOG(INFO) << "Edge layer gpu forwarding, with img_num: "<< top[0]->shape(0);
/*TODO:Uncomment this to visualize
//		int rand_id = rand() % 100000;
for (int n = 0; n < top[0]->shape(0); n++) {
for (int c = 0; c < top[0]->shape(1); c++){
std::stringstream ss_input;
ss_input << rand_id << "_" << top[0]->shape(3) << "_" << top[0]->shape(2) << "img_" << n << "_" << c << "_edge";
const int img_pos = n * top[0]->shape(3) * top[0]->shape(2) * top[0]->shape(1);
const int ch_pos = c * top[0]->shape(3) * top[0]->shape(2);
visualize_edge(top[0]->cpu_data() + img_pos + ch_pos, top[0]->shape(3), top[0]->shape(2), ss_input.str());
LOG(INFO) << "GPU CALL: LABEL SPAPE: " << top[0]->shape(0) << " " << top[0]->shape(1) << " " <<
		  top[0]->shape(2) << " " << top[0]->shape(3);
}

}
//*/
}

template<typename Dtype>
void EdgeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
this->layer->Backward(conv_bottom, propagate_down, conv_top);
}


INSTANTIATE_LAYER_GPU_FUNCS(EdgeLayer);
}  // namespace caffe
