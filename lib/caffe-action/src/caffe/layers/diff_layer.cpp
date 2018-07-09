//
// Created by kevin on 9/18/17.
//
#include "caffe/layers/diff_layer.hpp"

namespace caffe {
	template<typename Dtype>
	void DiffLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
		this->kernel_size = this->layer_param_.diff_param().kernel_size();
		CHECK_EQ(this->kernel_size % 2, 1); // kernel size must be odd
		next_max_ids.Reshape(bottom[1]->shape());
		if (!this->layer_param_.diff_param().is_flow())
			top[0]->Reshape(bottom[1]->shape());
		else
			top[0]->Reshape(bottom[1]->shape(0), bottom[1]->shape(1) * 2, bottom[1]->shape(2), bottom[1]->shape(3));

	}


	/*
	 * 1. Sliding over the whole bottom in channel axis.
	   2. Calculate the difference inside the kernel (What operation, find max diff?)
	   3. Sum all.
	 */
	template<typename Dtype>
	void DiffLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
		const Dtype *bottom_data = bottom[0]->cpu_data();
		const Dtype *bottom_next_data = bottom[1]->cpu_data();
		Dtype * top_data = top[0]->mutable_cpu_data();
		const int num = bottom[0]->num();
		const int channels = bottom[0]->channels();
		const int width = bottom[0]->width();
		const int height = bottom[0]->height();
		Dtype *next_max_id_data = this->next_max_ids.mutable_cpu_data();
		for (int n = 0; n < num; n++)
			for (int c = 0; c < channels; c++)
				for (int w = 0; w < width; w++)
					for (int h = 0; h < width; h++) {
						const int i = n * channels * width * height + c * width * height + w * height + h;
//						const int kernel_count = this->kernel_size * this->kernel_size;
						Dtype diff = Dtype(255);
						for (int m = 0; m < this->kernel_size; m++)
							for (int j = 0; j < this->kernel_size; j++) {
								const int w_off = w - (this->kernel_size - 1 / 2) + m;
								const int h_off = h - (this->kernel_size - 1 / 2) + j;
								if (w_off >= 0 && w_off < width && h_off >= 0 && h_off < height) {
									const int next_i = i - (this->kernel_size - 1) / 2 + m * j;
									Dtype temp_diff = bottom_data[i] - bottom_next_data[next_i];
									diff = abs(temp_diff) < diff ? temp_diff : diff;
									next_max_id_data[i] = next_i;
									top_data[i] = diff;
								} else {
									//auto-skip
									continue;
								}

							}
					}
	}

	template<typename Dtype>
	void DiffLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
								 const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
		NOT_IMPLEMENTED;
	}

#ifdef CPU_ONLY
	STUB_GPU(DiffLayer);
#endif

	INSTANTIATE_CLASS(DiffLayer);

	REGISTER_LAYER_CLASS(Diff);
} // namespace caffe