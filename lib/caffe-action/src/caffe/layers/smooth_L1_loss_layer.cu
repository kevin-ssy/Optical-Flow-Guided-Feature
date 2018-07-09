// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include "caffe/loss_layers.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace caffe {

	template<typename Dtype>
	void visualize(Dtype *data, int width, int height, string img_key) {
//		cv::Mat img1(cv::Size(width, height), CV_8UC3), img2(cv::Size(width, height), CV_8UC3);
		cv::Mat flow_x(cv::Size(width, height), CV_8UC1), flow_y(cv::Size(width, height), CV_8UC1);

		//flow_x
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++) {
				unsigned char &grey = flow_x.at<uchar>(y, x);
				grey = (unsigned char) (data[y * width + x]);
			}

		//flow_y
		int flow_y_offset = height * width;
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++) {
//				if(flow_y_offset + y * width + x % 100 ==0)
//				LOG(INFO) <<data[flow_y_offset + y * width + x];
				unsigned char &grey = flow_y.at<uchar>(y, x);
				grey = (unsigned char) (data[flow_y_offset + y * width + x]);
			}

		cv::imwrite("visualize/" + img_key + "_l1_flow_x.jpg", flow_x);
		cv::imwrite("visualize/" + img_key + "_l1_flow_y.jpg", flow_y);
		LOG(INFO) << "Img:" << img_key << " wrote.";

	}

template <typename Dtype>
__global__ void SmoothL1Forward(const int n, const Dtype* in, Dtype* out) {
  // f(x) = 0.5 * x^2    if |x| < 1
  //        |x| - 0.5    otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    if (abs_val < 1) {
      out[index] = 0.5 * val * val;
    } else {
      out[index] = abs_val - 0.5;
    }
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	/*TODO:Uncomment this to visualize
for(int n = 0; n < bottom[1]->num(); n++){
std::stringstream ss, ss_pred;
ss << bottom[1]->shape(3)<< "_" <<bottom[1]->shape(2)<< n << "_gt";
ss_pred << bottom[0]->shape(3)<< "_" <<bottom[0]->shape(2)<< n << "_pred";
visualize(bottom[1]->cpu_data(), bottom[1]->shape(3), bottom[1]->shape(2), ss.str());
visualize(bottom[0]->cpu_data(), bottom[0]->shape(3), bottom[0]->shape(2), ss_pred.str());
LOG(INFO) << "LABEL SPAPE: " << bottom[1]->shape(0)<< " " << bottom[1]->shape(1)<< " "<<
bottom[1]->shape(2)<<" "<< bottom[1]->shape(3);

}
*/
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());    // d := b0 - b1
  if (has_weights_) {
    caffe_gpu_mul(
        count,
        bottom[2]->gpu_data(),
        diff_.gpu_data(),
        diff_.mutable_gpu_data());  // d := w * (b0 - b1)
  }
  SmoothL1Forward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, diff_.gpu_data(), errors_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;

  Dtype loss;
  caffe_gpu_asum(count, errors_.gpu_data(), &loss);
  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
}

template <typename Dtype>
__global__ void SmoothL1Backward(const int n, const Dtype* in, Dtype* out) {
  // f'(x) = x         if |x| < 1
  //       = sign(x)   otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    if (abs_val < 1) {
      out[index] = val;
    } else {
      out[index] = (Dtype(0) < val) - (val < Dtype(0));
    }
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = diff_.count();
  SmoothL1Backward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, diff_.gpu_data(), diff_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                           // alpha
          diff_.gpu_data(),                // x
          Dtype(0),                        // beta
          bottom[i]->mutable_gpu_diff());  // y
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SmoothL1LossLayer);

}  // namespace caffe
