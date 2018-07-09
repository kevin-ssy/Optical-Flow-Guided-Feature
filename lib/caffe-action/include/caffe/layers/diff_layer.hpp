//
// Created by kevin on 9/18/17.
//


#ifndef CAFFE_DIFF_LAYER_HPP_
#define CAFFE_DIFF_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <vector>


namespace caffe {
	template <typename Dtype>
	class DiffLayer : public Layer<Dtype> {
	public:
		explicit DiffLayer(const LayerParameter& param)
				: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){}

		virtual inline const char* type() const { return "Diff"; }
		virtual inline int MinBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

//		cv::Mat fillImage(const Dtype *data, int width, int height, bool is_color);

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int kernel_size;
		Blob<Dtype> next_max_ids;


	};
}


#endif //CAFFE_DIFF_LAYER_HPP_
