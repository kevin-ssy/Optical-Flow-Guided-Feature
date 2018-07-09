//
// Created by kevin on 4/19/17.
//

#ifndef CAFFE_ROI_GENERATING_LAYER_HPP
#define CAFFE_ROI_GENERATING_LAYER_HPP
#include <string>
#include <utility>
#include <vector>

#include <boost/unordered_map.hpp>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
namespace caffe{


	template <typename Dtype>
	class ROIGeneratingLayer : public Layer<Dtype> {
	public:
		explicit ROIGeneratingLayer(const LayerParameter& param)
				: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ROIGenerating"; }

		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int MaxBottomBlobs() const { return 1; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline int MaxTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){};
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){};
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};

	};
} // namespace caffe



#endif //CAFFE_ROI_GENERATING_LAYER_HPP
