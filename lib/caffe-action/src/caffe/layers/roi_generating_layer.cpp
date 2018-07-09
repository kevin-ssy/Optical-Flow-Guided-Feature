////
//// Created by kevin on 4/19/17.
////
//
//#include "caffe/layers/roi_generating_layer.hpp"
//
//namespace caffe {
//
//
//	template<typename Dtype>
//	void ROIGeneratingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
//											   const vector<Blob<Dtype> *> &top) {
//		// 1. index matching
//		const Dtype* roi_data = bottom[0]->cpu_data();
//		const int width = bottom[0]->shape(3);
//		const int height = bottom[0]->shape(2);
//		const int batch_size = bottom[0]->shape(0);
//		top[0]->Reshape(batch_size, 1, 1, 5);
//		for (int item_id = 0; item_id < batch_size; item_id++){
//			top[0]->mutable_cpu_data()[item_id * 5] = Dtype(item_id);
//			for (int coor_id = 1; coor_id < 5; coor_id++)
//				top[0]->mutable_cpu_data()[item_id * 5 + coor_id] = roi_data[item_id * 5 + coor_id];
//		}
//
//	}
//
//	template<typename Dtype>
//	void ROIGeneratingLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
//											const vector<Blob<Dtype> *> &top) {
//	}
//
//
//	INSTANTIATE_CLASS(ROIGeneratingLayer);
//
//	REGISTER_LAYER_CLASS(ROIGenerating);
//} // namespace caffe