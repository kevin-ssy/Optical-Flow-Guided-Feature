#include "caffe/layers/diff_layer.hpp"


namespace caffe {
	template<typename Dtype>
	__global__ void CalcDiff(const int nthreads, const int num, const int channels, const int width, const int height,
							 const int kernel_size, const Dtype *bottom_data_a,
							 const Dtype *bottom_data_b, Dtype *next_max_id_data, Dtype *top_data, bool is_flow) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			const int w = index % width;
			const int h = index / width % height;
			const int c = index / width / height % channels;
			const int n = index / width / height / channels;
//			const int i = n * channels * width * height + c * width * height + w * height + h;
//			const int kernel_count = kernel_size * kernel_size;
			Dtype diff = Dtype(255);
			Dtype flow_x = Dtype(0);
			Dtype flow_y = Dtype(0);
			for (int m = - (kernel_size - 1) / 2; m <= (kernel_size - 1) / 2; m++)
				for (int j = - (kernel_size - 1) / 2; j <= (kernel_size - 1) / 2; j++) {
					const int w_off = w + m;
					const int h_off = h + j;
					if (w_off >= 0 && w_off < width && h_off >= 0 && h_off < height) {
						const int next_i = index + j * width + m;
						Dtype temp_diff = bottom_data_a[index] - bottom_data_b[next_i];
						if (abs(temp_diff) < abs(diff)) {
							diff = temp_diff;
							flow_x = Dtype(m) / Dtype((kernel_size - 1) / 2);
							flow_y = Dtype(j) / Dtype((kernel_size - 1) / 2);
							next_max_id_data[index] = next_i;
						} else if (abs(temp_diff) == abs(diff) && ((abs(m) < abs(flow_x)) || (abs(j) < abs(flow_y)))) {
							flow_x = Dtype(m) / Dtype((kernel_size - 1) / 2);
							flow_y = Dtype(j) / Dtype((kernel_size - 1) / 2);
							next_max_id_data[index] = next_i;
						}
					}
				}
			if (is_flow) {
				// index here refers to the channel X, another to the channel Y
				const int index_x = n * channels * height * width + 2 * c * height * width + h * width + w;
				const int index_y = n * channels * height * width + (2 * c + 1) * height * width + h * width + w;
				top_data[index_x] = flow_x;
				top_data[index_y] = flow_y;
			} else top_data[index] = diff;
		}
	}


	template<typename Dtype>
	__global__ void BackwardBottomNext(const int nthreads, const int num, const int channels, const int width,
									   const int height, const int kernel_size, Dtype *bottom_diff_next,
									   Dtype *next_max_id_data, Dtype *top_diff, bool is_flow) {
		if (!is_flow){
			CUDA_KERNEL_LOOP(index, nthreads) {
				const int next_index = static_cast<int>(next_max_id_data[index]);
				bottom_diff_next[next_index] = Dtype(-1) * top_diff[index];
			}
		}
		// No BP when setting in flow mode

	}

	template<typename Dtype>
	void DiffLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
		const Dtype *bottom_data = bottom[0]->gpu_data();
		const Dtype *bottom_next_data = bottom[1]->gpu_data();
		Dtype *next_max_id_data = this->next_max_ids.mutable_gpu_data();
		Dtype *top_data = top[0]->mutable_gpu_data();
		const int count = bottom[0]->count();
		const int num = bottom[0]->num();
		const int channels = bottom[0]->channels();
		const int width = bottom[0]->width();
		const int height = bottom[0]->height();
		bool is_flow = this->layer_param_.diff_param().is_flow();

		CalcDiff<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, num, channels, width, height,
				this->kernel_size, bottom_data, bottom_next_data, next_max_id_data, top_data, is_flow);
	}

	template<typename Dtype>
	void DiffLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
								 const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {

		if (propagate_down[0]) {
			const int count = bottom[0]->count();
			caffe_copy(count, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
		}
		if (propagate_down[1]) {
			Dtype *bottom_next_diff = bottom[1]->mutable_gpu_diff();
			Dtype *next_max_id_data = this->next_max_ids.mutable_gpu_data();
			Dtype *top_diff = top[0]->mutable_gpu_diff();
			const int count = bottom[1]->count();
			const int num = bottom[1]->num();
			const int channels = bottom[1]->channels();
			const int width = bottom[1]->width();
			const int height = bottom[1]->height();
			bool is_flow = this->layer_param_.diff_param().is_flow();
			BackwardBottomNext<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, num, channels, width,
					height, this->kernel_size, bottom_next_diff, next_max_id_data, top_diff,is_flow);
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(DiffLayer);
} // namespace caffe