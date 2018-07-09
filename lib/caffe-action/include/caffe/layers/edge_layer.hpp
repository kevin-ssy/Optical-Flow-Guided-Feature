//
// Created by kevin on 7/14/17.
//

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#ifndef CAFFE_EDGE_LAYER_HPP
#define CAFFE_EDGE_LAYER_HPP
namespace caffe {

	template <typename Dtype>
	class EdgeLayer : public Layer<Dtype> {
	public:
		explicit EdgeLayer(const LayerParameter& param)
				: Layer<Dtype>(param) {}

		virtual inline const char* type() const { return "Edge"; }

		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
							 const vector<Blob<Dtype>*>& top){}
		shared_ptr<Layer<Dtype> > layer;
		LayerParameter layer_param;
		vector<Blob<Dtype> *> conv_bottom;
		vector<Blob<Dtype> *> conv_top;

	protected:

		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	};

}  // namespace caffe

#endif //CAFFE_EDGE_LAYER_HPP
