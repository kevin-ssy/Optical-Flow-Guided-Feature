//
// Created by kevin on 3/6/17.
//

#ifndef CAFFE_ELTWISE_LAYER_HPP
#define CAFFE_ELTWISE_LAYER_HPP

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
	template<typename Dtype>
	class EltwiseLayer : public Layer<Dtype> {
	public:
		explicit EltwiseLayer(const LayerParameter &param)
				: Layer<Dtype>(param) {}

		virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
								const vector<Blob<Dtype> *> &top);

		virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
							 const vector<Blob<Dtype> *> &top);

		virtual inline const char *type() const { return "Eltwise"; }

		virtual inline int MinBottomBlobs() const { return 2; }

		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
								 const vector<Blob<Dtype> *> &top);

		virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
								 const vector<Blob<Dtype> *> &top);

		virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
								  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

		virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
								  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

		EltwiseParameter_EltwiseOp op_;
		vector<Dtype> coeffs_;
		Blob<int> max_idx_;
		Blob<Dtype> rng_buffer_;

		bool stable_prod_grad_;
	};
} // namespace caffe

#endif //CAFFE_ELTWISE_LAYER_HPP
