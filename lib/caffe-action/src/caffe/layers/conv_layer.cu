#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace caffe {

	template<typename Dtype>
	void visualize(const Dtype *data, int num, int channels, int width, int height, string img_key) {
		cv::Mat img1(cv::Size(width, height), CV_8UC3), img2(cv::Size(width, height), CV_8UC3);
		LOG(INFO) << "Visualizing: " << channels << " " << width << " " << height;
//      cv::Mat flow_x(cv::Size(width, height), CV_8UC1), flow_y(cv::Size(width, height), CV_8UC1);
		if (channels == 6 && width == 256 && height == 256) {
			for (int n = 0; n < num; n++) {
				//img1
				for (int c = 0; c < 3; c++)
					for (int y = 0; y < height; y++)
						for (int x = 0; x < width; x++) {
							cv::Vec3b &rgb = img1.at<cv::Vec3b>(y, x);
							rgb[c] = (unsigned char) data[n * 6 * height * width + c * height * width + y * width + x];
						}
				//img2
				int img2_offset = n * 6 * height * width + 3 * height * width;
				for (int c = 0; c < 3; c++)
					for (int y = 0; y < height; y++)
						for (int x = 0; x < width; x++) {
							cv::Vec3b &rgb = img2.at<cv::Vec3b>(y, x);
							rgb[c] = (unsigned char) data[img2_offset + c * height * width + y * width + x];
						}

				std::stringstream ss_img1;
				std::stringstream ss_img2;
				ss_img1 << "visualize/" << img_key << "_" << n << "_conv_img1.jpg";
				ss_img2 << "visualize/" << img_key << "_" << n << "_conv_img2.jpg";
				cv::imwrite(ss_img1.str(), img1);
				cv::imwrite(ss_img2.str(), img2);
				LOG(INFO) << "Img:" << img_key << " wrote.";
			}
		}

	}

	template<typename Dtype>
	void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
											  const vector<Blob<Dtype> *> &top) {
//		LOG(INFO) << "Visualizing0: " <<bottom[0]->shape(0) << " " << bottom[0]->shape(1)
//				  << " " << bottom[0]->shape(2) << " " << bottom[0]->shape(3);

  /*TODO: Uncomment this to visualize

//		ss << "from_concat" << rand() % 1000;
		std::stringstream ss;
		ss << bottom[0]->shape(3) << "_" << bottom[0]->shape(2) << "_tsn";

		visualize(bottom[0]->cpu_data(), bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(3),
				  bottom[0]->shape(2), ss.str());

*/


		const Dtype *weight = this->blobs_[0]->gpu_data();
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype *bottom_data = bottom[i]->gpu_data();
			Dtype *top_data = top[i]->mutable_gpu_data();
			for (int n = 0; n < this->num_; ++n) {
				this->forward_gpu_gemm(bottom_data + bottom[i]->offset(n), weight,
									   top_data + top[i]->offset(n));
				if (this->bias_term_) {
					const Dtype *bias = this->blobs_[1]->gpu_data();
					this->forward_gpu_bias(top_data + top[i]->offset(n), bias);
				}
			}
		}
	}

	template<typename Dtype>
	void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
											   const vector<bool> &propagate_down,
											   const vector<Blob<Dtype> *> &bottom) {
		const Dtype *weight = this->blobs_[0]->gpu_data();
		Dtype *weight_diff = this->blobs_[0]->mutable_gpu_diff();
		for (int i = 0; i < top.size(); ++i) {
			const Dtype *top_diff = top[i]->gpu_diff();
			// Bias gradient, if necessary.
			if (this->bias_term_ && this->param_propagate_down_[1]) {
				Dtype *bias_diff = this->blobs_[1]->mutable_gpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					this->backward_gpu_bias(bias_diff, top_diff + top[i]->offset(n));
				}
			}
			if (this->param_propagate_down_[0] || propagate_down[i]) {
				const Dtype *bottom_data = bottom[i]->gpu_data();
				Dtype *bottom_diff = bottom[i]->mutable_gpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					// gradient w.r.t. weight. Note that we will accumulate diffs.
					if (this->param_propagate_down_[0]) {
						this->weight_gpu_gemm(bottom_data + bottom[i]->offset(n),
											  top_diff + top[i]->offset(n), weight_diff);
					}
					// gradient w.r.t. bottom data, if necessary.
					if (propagate_down[i]) {
						this->backward_gpu_gemm(top_diff + top[i]->offset(n), weight,
												bottom_diff + bottom[i]->offset(n));
					}
				}
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
