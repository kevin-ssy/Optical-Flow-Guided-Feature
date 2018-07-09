#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void Concat(const int nthreads, const Dtype* in_data,
						   const bool forward, const int num_concats, const int concat_size,
						   const int top_concat_axis, const int bottom_concat_axis,
						   const int offset_concat_axis, Dtype* out_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			const int total_concat_size = concat_size * bottom_concat_axis;
			const int concat_num = index / total_concat_size;
			const int concat_index = index % total_concat_size;
			const int top_index = concat_index +
								  (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
			if (forward) {
				out_data[top_index] = in_data[index];
			} else {
				out_data[index] = in_data[top_index];
			}
		}
	}

	template<typename Dtype>
	void writeDataBlob(const Dtype *data, int num, int channels, int width, int height, string img_key){
		if (channels == 6 && width == 256 && height == 256) {
			Dtype output[num * 6 * 256 * 256];
			for (int n = 0; n < num; n++) {
				for (int c = 0; c < channels; c++)
					for (int y = 0; y < height; y++)
						for (int x = 0; x < width; x++) {
							int data_id = n * 6 * height * width + c * height * width + y * width + x;
							output[data_id] = data[data_id];
						}
			}
			std::stringstream ss_blob;
			ss_blob << "visualize/" << img_key << "_input_reshape_blob.blob";
			FILE *fblob = fopen(ss_blob.str().c_str(), "w");
			if (fblob != NULL){
				fwrite(output, sizeof(Dtype), num * 6 * 256 * 256, fblob);

			}else{
				LOG(INFO)<< "Cannot open file" << ss_blob.str();
			}
			fclose(fblob);
			LOG(INFO) << "Blob:" << img_key << " wrote.";
		}
	}

	template<typename Dtype>
	void writeLabelBlob(const Dtype *data, int num, int channels, int width, int height, string img_key){
		if (channels == 2 && width == 256 && height == 256) {
			Dtype output[num * 2 * 256 * 256];
			for (int n = 0; n < num; n++) {
				for (int c = 0; c < channels; c++)
					for (int y = 0; y < height; y++)
						for (int x = 0; x < width; x++) {
							int data_id = n * 2 * height * width + c * height * width + y * width + x;
							output[data_id] = data[data_id];
						}
			}
			std::stringstream ss_blob;
			ss_blob << "visualize/" << img_key << "_label_reshape_blob.blob";
			FILE *fblob = fopen(ss_blob.str().c_str(), "w");
			if (fblob != NULL){
				fwrite(output, sizeof(Dtype), num * 2 * 256 * 256, fblob);

			}else{
				LOG(INFO)<< "Cannot open file" << ss_blob.str();
			}
			fclose(fblob);
			LOG(INFO) << "Blob:" << img_key << " wrote.";
		}
	}


	template<typename Dtype>
	void visualize(const Dtype *data, int num, int channels, int width, int height, string img_key) {
		cv::Mat img1(cv::Size(width, height), CV_8UC3), img2(cv::Size(width, height), CV_8UC3);
//      cv::Mat flow_x(cv::Size(width, height), CV_8UC1), flow_y(cv::Size(width, height), CV_8UC1);
		if (channels == 6 && width == 256 && height == 256) {
			for (int n = 0; n < num; n++) {
				//img1
				for (int c = 0; c < 3; c++)
					for (int y = 0; y < height; y++)
						for (int x = 0; x < width; x++) {
							cv::Vec3b &rgb = img1.at<cv::Vec3b>(y, x);
//							rgb[c] = (unsigned char) data[n * 6 * height * width + c * height * width + y * width + x];
							if (c == 0) rgb[c] = (unsigned char) (data[n * 6 * height * width + c * height * width + y * width + x]+ 104);//+ 104
							if (c == 1) rgb[c] = (unsigned char) (data[n * 6 * height * width + c * height * width + y * width + x]+ 117);//+ 117
							if (c == 2) rgb[c] = (unsigned char) (data[n * 6 * height * width + c * height * width + y * width + x]+ 123);//+ 123
						}
				//img2
				int img2_offset = n * 6 * height * width + 3 * height * width;
				for (int c = 0; c < 3; c++)
					for (int y = 0; y < height; y++)
						for (int x = 0; x < width; x++) {
							cv::Vec3b &rgb = img2.at<cv::Vec3b>(y, x);
//							rgb[c] = (unsigned char) data[img2_offset + c * height * width + y * width + x];
							if (c == 0) rgb[c] = (unsigned char) (data[img2_offset + c * height * width + y * width + x]+ 104);//+ 104
							if (c == 1) rgb[c] = (unsigned char) (data[img2_offset + c * height * width + y * width + x]+ 117);//+ 117
							if (c == 2) rgb[c] = (unsigned char) (data[img2_offset + c * height * width + y * width + x]+ 123);//+ 123
						}

				std::stringstream ss_img1;
				std::stringstream ss_img2;
				ss_img1 << "visualize/" << img_key << n << "_input_reshape_img1.jpg";
				ss_img2 << "visualize/" << img_key << n << "_input_reshape_img2.jpg";
				cv::imwrite(ss_img1.str(), img1);
				cv::imwrite(ss_img2.str(), img2);
				LOG(INFO) << "Img:" << img_key << " wrote.";
			}
		}

	}

	template<typename Dtype>
	void visualize_flow(const Dtype *data, int num, int channels, int width, int height, string img_key) {
		cv::Mat img1(cv::Size(width, height), CV_8UC1), img2(cv::Size(width, height), CV_8UC1);
//      cv::Mat flow_x(cv::Size(width, height), CV_8UC1), flow_y(cv::Size(width, height), CV_8UC1);
		if (channels == 2 && width == 256 && height == 256) {
			for (int n = 0; n < num; n++) {
				//img1
//				for (int c = 0; c < 3; c++)
					for (int y = 0; y < height; y++)
						for (int x = 0; x < width; x++) {
							unsigned char &grey = img1.at<uchar>(y, x);
							grey = (unsigned char) data[n * 2 *height * width + y * width + x];
						}
				//img2
				int img2_offset = n * 2 * height * width + height * width;
					for (int y = 0; y < height; y++)
						for (int x = 0; x < width; x++) {
							unsigned char &grey = img2.at<uchar>(y, x);
							grey = (unsigned char) data[img2_offset + y * width + x];
						}

				std::stringstream ss_img1;
				std::stringstream ss_img2;
				ss_img1 << "visualize/" << img_key << n << "_input_reshape_flow_x.jpg";
				ss_img2 << "visualize/" << img_key << n << "_input_reshape_flow_y.jpg";
				cv::imwrite(ss_img1.str(), img1);
				cv::imwrite(ss_img2.str(), img2);
				LOG(INFO) << "Img:" << img_key << " wrote.";
			}
		}

	}

	template<typename Dtype>
	void ConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
										 const vector<Blob<Dtype> *> &top) {
		if (bottom.size() == 1) { return; }
		Dtype* top_data = top[0]->mutable_gpu_data();
		int offset_concat_axis = 0;
		const int top_concat_axis = top[0]->shape(concat_axis_);
		const bool kForward = true;
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->gpu_data();
			const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
			const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
			const int nthreads = bottom_concat_size * num_concats_;
			Concat<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
					<<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
					nthreads, bottom_data, kForward, num_concats_, concat_input_size_,
							top_concat_axis, bottom_concat_axis, offset_concat_axis, top_data);
			offset_concat_axis += bottom_concat_axis;
		}
  /*TODO: Uncomment this to visualize

//		ss << "from_concat" << rand() % 1000;
		std::stringstream ss;
		ss << top[0]->shape(3) << "_" << top[0]->shape(2) << "_";
		visualize(top[0]->cpu_data(), top[0]->shape(0), top[0]->shape(1), top[0]->shape(3), top[0]->shape(2),
				  ss.str());
		writeDataBlob(top[0]->cpu_data(), top[0]->shape(0), top[0]->shape(1), top[0]->shape(3), top[0]->shape(2),
				  ss.str());
		writeLabelBlob(top[0]->cpu_data(), top[0]->shape(0), top[0]->shape(1), top[0]->shape(3), top[0]->shape(2),
					   ss.str());
		visualize_flow(top[0]->cpu_data(), top[0]->shape(0), top[0]->shape(1), top[0]->shape(3), top[0]->shape(2),
				  ss.str());

 */
	}

	template <typename Dtype>
	void ConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
										  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (bottom.size() == 1) { return; }
		const Dtype* top_diff = top[0]->gpu_diff();
		int offset_concat_axis = 0;
		const int top_concat_axis = top[0]->shape(concat_axis_);
		const bool kForward = false;
		for (int i = 0; i < bottom.size(); ++i) {
			const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
			if (propagate_down[i]) {
				Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
				const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
				const int nthreads = bottom_concat_size * num_concats_;
				Concat<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
						<<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
						nthreads, top_diff, kForward, num_concats_, concat_input_size_,
								top_concat_axis, bottom_concat_axis, offset_concat_axis, bottom_diff);
			}
			offset_concat_axis += bottom_concat_axis;
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ConcatLayer);

}  // namespace caffe
