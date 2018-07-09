#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/l1_loss_layer.hpp"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "caffe/layers/custom_data_layer.hpp"
//#include "../../../../../../../usr/include/c++/4.9/iosfwd"

using namespace std;
namespace caffe {


	template<typename Dtype>
	void visualize(const Dtype *data, int width, int height, string img_key) {
//		cv::Mat img1(cv::Size(width, height), CV_8UC3), img2(cv::Size(width, height), CV_8UC3);
		cv::Mat flow_x(cv::Size(width, height), CV_8UC1), flow_y(cv::Size(width, height), CV_8UC1);

		//flow_x
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++) {
				unsigned char &grey = flow_x.at<uchar>(y, x);
				grey = (unsigned char) (data[y * width + x] * 20);
			}

		//flow_y
		int flow_y_offset = height * width;
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++) {
//				if(flow_y_offset + y * width + x % 100 ==0)
//				LOG(INFO) <<data[flow_y_offset + y * width + x];
				unsigned char &grey = flow_y.at<uchar>(y, x);
				grey = (unsigned char) (data[flow_y_offset + y * width + x] * 20);
			}

		cv::imwrite("visualize/" + img_key + "_l1_flow_x.jpg", flow_x);
		cv::imwrite("visualize/" + img_key + "_l1_flow_y.jpg", flow_y);
		LOG(INFO) << "Img:" << img_key << " wrote.";

	}

	template<typename Dtype>
	void visualize_flownet(const Dtype *data, int width, int height, string img_key) {
		cv::Mat flow_x(cv::Size(width, height), CV_8UC1), flow_y(cv::Size(width, height), CV_8UC1);
#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255 * ((v) - (L)) / ((H)-(L))))
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++) {
				unsigned char &grey = flow_x.at<uchar>(y, x);
				grey = (unsigned char) CAST(data[y * width + x], -40, 40);
			}
		int flow_y_offset = height * width;
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++) {
//				if(flow_y_offset + y * width + x % 100 ==0)
//				LOG(INFO) <<data[flow_y_offset + y * width + x];
				unsigned char &grey = flow_y.at<uchar>(y, x);
				grey = (unsigned char) CAST(data[flow_y_offset + y * width + x], -40, 40);
			}
#undef CAST
		cv::imwrite("visualize/" + img_key + "_l1_flow_x.jpg", flow_x);
		cv::imwrite("visualize/" + img_key + "_l1_flow_y.jpg", flow_y);
		LOG(INFO) << "Img:" << img_key << " wrote.";
	}

	template<typename Dtype>
	__global__ void ComputeSign(const int n, const Dtype *in, Dtype *out) {
		CUDA_KERNEL_LOOP(index, n) {
			out[index] = in[index] > 0 ? Dtype(1) : Dtype(-1);
		}
	}

// TODO maybe change the way of detecting NaNs

	template<typename Dtype>
	__global__ void FindNotNaNs(const int n, const Dtype *in, Dtype *out) {
		CUDA_KERNEL_LOOP(index, n) {
			out[index] = in[index] == in[index] ? Dtype(1) : Dtype(0);
		}
	}

	template<typename Dtype>
	__global__ void KillNaNs(const int n, const Dtype *in, Dtype *out) {
		CUDA_KERNEL_LOOP(index, n) {
			out[index] = in[index] == in[index] ? in[index] : Dtype(0);
		}
	}

	template<typename Dtype>
	__global__ void KillMasked(const int n, const Dtype *in, Dtype *out) {
		CUDA_KERNEL_LOOP(index, n) {
			out[index] = in[index] > Dtype(0.5) ? out[index] : Dtype(0);
//     out[index] = out[index]==out[index] ? out[index] : Dtype(0);
//     out[index] = out[index]>1e3 ? 0 : out[index];
//     out[index] = out[index]<-1e3 ? 0 : out[index];
		}
	}

	template<typename Dtype>
	__global__ void KillMaskedAcrossChannels(const int n, const int width_height, const Dtype *in, Dtype *out) {
		CUDA_KERNEL_LOOP(index, n) {
			const int mask_idx = index % width_height;
			out[index] = in[mask_idx] > Dtype(0.5) ? out[index] : Dtype(0);
		}
	}

	template<typename Dtype>
	__global__ void MaskPlateauValues(const int n, const Dtype *in, Dtype *out, Dtype plateau) {
		CUDA_KERNEL_LOOP(index, n) {
			if (fabs(in[index]) < plateau) out[index] = Dtype(0); // Mask out plateau values and keep other as is
		}
	}

	template<typename Dtype>
	__global__ void MaskPlateauValuesInitial(const int n, const Dtype *in, Dtype *out, Dtype plateau) {
		CUDA_KERNEL_LOOP(index, n) {
			out[index] = (fabs(in[index]) < plateau) ? Dtype(0) : Dtype(1);
		}
	}


	template<typename Dtype>
	void L1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
										 const vector<Blob<Dtype> *> &top) {
///*TODO:Uncomment this to visualize
		for(int n = 0; n < bottom[1]->num(); n++){
			std::stringstream ss, ss_pred;
			int id = int(rand() % 10000);
			ss << bottom[1]->shape(3)<< "_" <<bottom[1]->shape(2)<< n << "_gt_tsn_" << id;
			ss_pred << bottom[0]->shape(3)<< "_" <<bottom[0]->shape(2)<< n << "_pred_tsn_"<< id;
			visualize(bottom[1]->cpu_data(), bottom[1]->shape(3), bottom[1]->shape(2), ss.str());
			visualize_flownet(bottom[0]->cpu_data(), bottom[0]->shape(3), bottom[0]->shape(2), ss_pred.str());
			LOG(INFO) << "LABEL SPAPE: " << bottom[1]->shape(0)<< " " << bottom[1]->shape(1)
					  << " "<<bottom[1]->shape(2)<<" "<< bottom[1]->shape(3);
		}
//*/

//	int counter = 0;
//	for (int i=0; i<bottom[0]->count(); i++){
//
////	LOG(INFO) << "Data: " << bottom[0]->cpu_data()[i];
////	LOG(INFO) << "Label: " << bottom[1]->cpu_data()[i];
////	LOG(INFO) << "Loss: " << bottom[0]->cpu_data()[i] - bottom[1]->cpu_data()[i];
//	if ( bottom[1]->cpu_data()[i] == 0) counter++;
//
//}
//LOG(INFO)<< counter << " zeros in label";

		Blob<Dtype> *diffptr = diff_top_vec_[0];

		Dtype dot, loss;
		if (bottom.size() > 1) {
			diff_layer_->Forward(bottom, diff_top_vec_);
		}


// if necessary, compute the number of not-NaNs
		int count = bottom[0]->count();
		int num = bottom[0]->num();
		FindNotNaNs<Dtype> << < CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
				count, diffptr->gpu_data(), mask_.mutable_gpu_data());
		cudaDeviceSynchronize();
		CUDA_POST_KERNEL_CHECK;
//		for (int i = 0; i < mask_.count(); i++){
//			if(i%100==0) LOG(INFO)<<"mask: "<< mask_.cpu_data()[i];
//		}



		if (this->layer_param_.l1_loss_param().normalize_by_num_entries()) {
			caffe_gpu_dot(count, mask_.gpu_data(), mask_.gpu_data(), &normalize_coeff_);
			normalize_coeff_ /= mask_.channels();
		} else {
			normalize_coeff_ = num;
		}
//        LOG(INFO)<<"normalize_coeff_"<<normalize_coeff_;
		if (this->layer_param_.l1_loss_param().l2_per_location()) {
// set masked (NaNs only) to zero
			KillMasked<Dtype> << < CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
					count, mask_.gpu_data(), diffptr->mutable_gpu_data());
			cudaDeviceSynchronize();
			CUDA_POST_KERNEL_CHECK;

			square_layer_->Forward(diff_top_vec_, square_top_vec_);
			sum_layer_->Forward(square_top_vec_, sum_top_vec_);

// Mask plateau in summed blob (only one channel):
			if (this->layer_param_.l1_loss_param().plateau() > 0) {
				float plateau_val_squared =
						this->layer_param_.l1_loss_param().plateau() * this->layer_param_.l1_loss_param().plateau();
				MaskPlateauValuesInitial<Dtype> << < CAFFE_GET_BLOCKS(sum_output_.count()), CAFFE_CUDA_NUM_THREADS >> >
																							(
																									sum_output_.count(), sum_output_.gpu_data(), plateau_l2_.mutable_gpu_data(), plateau_val_squared);
//				LOG(INFO)<<"plateau_val_squared: "<<plateau_val_squared;
				cudaDeviceSynchronize();
				CUDA_POST_KERNEL_CHECK;

				KillMasked<Dtype> << < CAFFE_GET_BLOCKS(sum_output_.count()), CAFFE_CUDA_NUM_THREADS >> > (
						sum_output_.count(), plateau_l2_.gpu_data(), sum_output_.mutable_gpu_data());
				cudaDeviceSynchronize();
				CUDA_POST_KERNEL_CHECK;
			}

			sqrt_layer_->Forward(sum_top_vec_, sqrt_top_vec_);
// Note sign_ is set to all ones in Reshape
			caffe_gpu_dot(sqrt_output_.count(), sqrt_output_.gpu_data(), sign_.gpu_data(), &dot);
		} else {
// Mask plateau:
			if (this->layer_param_.l1_loss_param().plateau() > 0) {
				MaskPlateauValues<Dtype> << < CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
						count, diffptr->gpu_data(), mask_.mutable_gpu_data(), this->layer_param_.l1_loss_param().plateau());
				CUDA_POST_KERNEL_CHECK;
			}

//mask_.print("MASK2");

// set masked (NaNs, plateau) to zero
			KillMasked<Dtype> << < CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
					count, mask_.gpu_data(), diffptr->mutable_gpu_data());
			CUDA_POST_KERNEL_CHECK;

			ComputeSign<Dtype> << < CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
					count, diffptr->gpu_data(), sign_.mutable_gpu_data());
			CUDA_POST_KERNEL_CHECK;
			caffe_gpu_dot(count, diffptr->gpu_data(), sign_.gpu_data(), &dot);
		}

//		for(int i = 0; i < diffptr->count(); i++){
//			LOG(INFO) <<"L1 diff "<< i << "/" << diffptr->count() - 1 << " : " << diffptr->cpu_data()[i];
//		}
//		LOG(INFO)<<"dot: "<<dot<<" "<<normalize_coeff_;
		loss = dot / normalize_coeff_;
		top[0]->mutable_cpu_data()[0] = loss;
	}

	template<typename Dtype>
	void L1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
										  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
		bool prop_down = propagate_down[0];
		if (bottom.size() > 1) prop_down |= propagate_down[1];

		Blob<Dtype> *diffptr = diff_top_vec_[0];

		if (prop_down) {
			const Dtype alpha = top[0]->cpu_diff()[0] / normalize_coeff_;
			if (this->layer_param_.l1_loss_param().l2_per_location()) {
				vector<bool> prop_down(1, true);
				caffe_gpu_axpby(sqrt_output_.count(), alpha, sign_.gpu_data(),
								Dtype(0), sqrt_output_.mutable_gpu_diff());
				sqrt_layer_->Backward(sqrt_top_vec_, prop_down, sum_top_vec_);

				if (this->layer_param_.l1_loss_param().plateau() > 0) {
					KillMasked<Dtype> << < CAFFE_GET_BLOCKS(sum_output_.count()), CAFFE_CUDA_NUM_THREADS >> > (
							sum_output_.count(), plateau_l2_.gpu_data(), sum_output_.mutable_gpu_diff());
					cudaDeviceSynchronize();
					CUDA_POST_KERNEL_CHECK;
				}

				sum_layer_->Backward(sum_top_vec_, prop_down, square_top_vec_);
				square_layer_->Backward(square_top_vec_, prop_down, diff_top_vec_);


			} else {
				caffe_gpu_axpby(diffptr->count(), alpha, sign_.gpu_data(),
								Dtype(0), diffptr->mutable_gpu_diff());
			}

			KillMasked<Dtype> << < CAFFE_GET_BLOCKS(diffptr->count()), CAFFE_CUDA_NUM_THREADS >> > (
					diffptr->count(), mask_.gpu_data(), diffptr->mutable_gpu_diff());
			CUDA_POST_KERNEL_CHECK;

			if (bottom.size() > 1) {
				diff_layer_->Backward(diff_top_vec_, propagate_down, bottom);
			}
		}

	}

	INSTANTIATE_LAYER_GPU_FUNCS(L1LossLayer);

}  // namespace caffe
