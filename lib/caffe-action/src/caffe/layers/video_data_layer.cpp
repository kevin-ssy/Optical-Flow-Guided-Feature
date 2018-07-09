#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#ifdef USE_MPI

#include "mpi.h"
#include <boost/filesystem.hpp>

using namespace boost::filesystem;
#endif

namespace caffe {
	template<typename Dtype>
	VideoDataLayer<Dtype>::~VideoDataLayer<Dtype>() {
		this->JoinPrefetchThread();
	}

	template<typename Dtype>
	void VideoDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
		const int new_height = this->layer_param_.video_data_param().new_height();
		const int new_width = this->layer_param_.video_data_param().new_width();
		const int new_length = this->layer_param_.video_data_param().new_length();
		const int num_segments = this->layer_param_.video_data_param().num_segments();
		const string &source = this->layer_param_.video_data_param().source();



		LOG(INFO) << "Opening file: " << source;
		std::ifstream infile(source.c_str());
		string filename;
		int label;
		int length;
		while (infile >> filename >> length >> label) {
			lines_.push_back(std::make_pair(filename, label));
			lines_duration_.push_back(length);
		}
		if (this->layer_param_.video_data_param().shuffle()) {
			const unsigned int prefectch_rng_seed = caffe_rng_rand();
			prefetch_rng_1_.reset(new Caffe::RNG(prefectch_rng_seed));
			prefetch_rng_2_.reset(new Caffe::RNG(prefectch_rng_seed));
			ShuffleVideos();
		}

		LOG(INFO) << "A total of " << lines_.size() << " videos.";
		lines_id_ = 0;

		//check name patter
		if (this->layer_param_.video_data_param().name_pattern() == "") {
			if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_RGB) {
				name_pattern_ = "image_%04d.jpg";
			} else if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW) {
				name_pattern_ = "flow_%c_%04d.jpg";
			}
		} else {
			name_pattern_ = this->layer_param_.video_data_param().name_pattern();
		}

		Datum datum;
		const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
		frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));
		int average_duration = 0;
		if (this->phase_ == TRAIN) {
			if (num_segments != 0)
				average_duration = (int) lines_duration_[lines_id_] / num_segments; // average duration of each segment
			else
				LOG(ERROR) << "num_segments is 0";
		} else {
			if (num_segments != 0)
				average_duration = (int) lines_duration_[lines_id_] / num_segments; // average duration of each segment
			else
				LOG(ERROR) << "num_segments is 0";
		}

		vector<int> offsets;
		if (this->phase_ == TRAIN) {
			for (int i = 0; i < num_segments; ++i) {
				caffe::rng_t *frame_rng = static_cast<caffe::rng_t *>(frame_prefetch_rng_->generator()); // randomly select a frame
				int offset = 0;
				if (average_duration - new_length + 1 != 0)
					offset = (*frame_rng)() %
							 (average_duration - new_length + 1); // ensure the frame is at the begining of a segment
				else
					LOG(ERROR) << "average_duration - new_length + 1 = 0!";
				offsets.push_back(offset + i * average_duration);
			}
		} else {
			for (int i = 0; i < num_segments; ++i) {
				if (average_duration >= new_length)
					offsets.push_back(int((average_duration - new_length + 1) / 2 + i * average_duration));
				else
					offsets.push_back(0);
			}
		}


		bool roi_pool_flag = this->layer_param_.video_data_param().roi_pool_flag();
		if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW)
			CHECK(ReadSegmentFlowToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
										 offsets, new_height, new_width, new_length, &datum, name_pattern_.c_str()));
		else if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_RGB)
			CHECK(ReadSegmentRGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
										offsets, new_height, new_width, new_length, &datum, true,
										name_pattern_.c_str()));
		else if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_RGB_FLOW) {
			CHECK(ReadSegmentRGBFlowToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
											offsets, new_height, new_width, new_length, &datum, true,
											name_pattern_.c_str()));
		} else if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_RGB_POSE) {
			if (this->layer_param_.video_data_param().has_img_list_path()) {
				LoadJoints(this->layer_param_.video_data_param().img_list_path(), this->person_map,
						   this->layer_param_.video_data_param().img_list_prefix());
				if (this->layer_param_.video_data_param().select_joints_size() > 0) {
					for (int s = 0; s < this->layer_param_.video_data_param().select_joints_size(); s++)
						select_joints.push_back(this->layer_param_.video_data_param().select_joints(s));
				}
				int roi_w = this->layer_param_.video_data_param().roi_w();
				int roi_h = this->layer_param_.video_data_param().roi_h();
				CHECK(ReadSegmentRGBPoseToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets,
												new_height, new_width, new_length, &datum, true, roi_pool_flag,
												name_pattern_.c_str(), this->person_map, 0, select_joints,
												this->layer_param_.video_data_param().stride(), roi_w, roi_h));
			} else {
				LOG(ERROR) << "img_list_path not set when setting to RGB_POSE modality";
			}

		}
		const int crop_size = this->layer_param_.transform_param().crop_size();
		const int crop_height = this->layer_param_.transform_param().crop_height();
		const int crop_width = this->layer_param_.transform_param().crop_width();
		const int batch_size = this->layer_param_.video_data_param().batch_size();
		const int num_branches = this->layer_param_.transform_param().is_roi_size();
//	LOG(INFO) << "CROP FLAG:" << crop_height > 0 && crop_width > 0;
		if (crop_height > 0 && crop_width > 0) {
			top[0]->Reshape(batch_size, num_branches * datum.channels(), crop_height, crop_width);
			this->prefetch_data_.Reshape(batch_size, num_branches * datum.channels(), crop_height, crop_width);
			this->transformed_data_.Reshape(1, num_branches * datum.channels(), crop_height, crop_width);
		} else if (crop_size > 0) {
			top[0]->Reshape(batch_size, num_branches * datum.channels(), crop_size, crop_size);
			this->prefetch_data_.Reshape(batch_size, num_branches * datum.channels(), crop_size, crop_size);
			this->transformed_data_.Reshape(1, datum.channels(), crop_size, crop_size);
		} else {
			top[0]->Reshape(batch_size, num_branches * datum.channels(), datum.height(), datum.width());
			this->prefetch_data_.Reshape(batch_size, num_branches * datum.channels(), datum.height(), datum.width());
			this->transformed_data_.Reshape(1, num_branches * datum.channels(), datum.height(), datum.width());
		}
		LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height()
				  << "," << top[0]->width();

		top[1]->Reshape(batch_size, 1, 1, 1); // label
		this->prefetch_label_.Reshape(batch_size, 1, 1, 1);

//	vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);

	}

	template<typename Dtype>
	void VideoDataLayer<Dtype>::ShuffleVideos() {
		caffe::rng_t *prefetch_rng1 = static_cast<caffe::rng_t *>(prefetch_rng_1_->generator());
		caffe::rng_t *prefetch_rng2 = static_cast<caffe::rng_t *>(prefetch_rng_2_->generator());
		shuffle(lines_.begin(), lines_.end(), prefetch_rng1);
		shuffle(lines_duration_.begin(), lines_duration_.end(), prefetch_rng2);
	}

	template<typename Dtype>
	void VideoDataLayer<Dtype>::InternalThreadEntry() {

		Datum datum;
		CHECK(this->prefetch_data_.count());
		Dtype *top_data = this->prefetch_data_.mutable_cpu_data();
		Dtype *top_label = this->prefetch_label_.mutable_cpu_data();
		VideoDataParameter video_data_param = this->layer_param_.video_data_param();
		const int batch_size = video_data_param.batch_size();
		const int new_height = video_data_param.new_height();
		const int new_width = video_data_param.new_width();
		const int new_length = video_data_param.new_length();
		const int num_segments = video_data_param.num_segments();
		const int lines_size = lines_.size();
		if (this->phase_ == TEST && this->layer_param_.video_data_param().has_video_id_path()) {
			string video_id_path = this->layer_param_.video_data_param().video_id_path();
			std::ifstream fvideo_id(video_id_path.c_str());
			string video_id_str;
			getline(fvideo_id, video_id_str);
			lines_id_ = atoi(video_id_str.c_str());
			fvideo_id.close();

//		LOG(INFO) << "Worker " << Caffe::device_id() << " lines_id_: "<< lines_id_;
		}

		for (int item_id = 0; item_id < batch_size; ++item_id) {
			CHECK_GT(lines_size, lines_id_);
			vector<int> offsets;
			int num_offsets = num_segments;
			if (video_data_param.has_num_offsets()) num_offsets = video_data_param.num_offsets();
			int average_duration = (int) (lines_duration_[lines_id_] - new_length) / (num_offsets - 1);

			if (this->phase_ == TRAIN && video_data_param.random_sample()) {
				int offset_id_seed = num_offsets > num_segments ? rand() % (num_offsets - num_segments - 1) : 0;
				for (int i = offset_id_seed; i < offset_id_seed + num_segments; ++i) {
					if (average_duration > 0) {
						caffe::rng_t *frame_rng = static_cast<caffe::rng_t *>(frame_prefetch_rng_->generator());//random generator
						int offset = average_duration - new_length > 0 ? (*frame_rng)() % (average_duration - new_length + 1) : 1; // frame id by 0
						//					LOG(INFO) << "Average duration: " << average_duration;
						//					LOG(INFO) << "offset: " << offset;
						//					LOG(INFO) << "Pushing " << offset + i * average_duration <<" into offsets!";
						offsets.push_back(std::min(offset + i * average_duration, lines_duration_[lines_id_] - new_length)); // segment position in video
					} else {
						offsets.push_back(0);
					}
				}
			} else {

				for (int i = 0; i < num_segments; ++i) {

					if (average_duration > 0)
					offsets.push_back(std::min(1 + i * average_duration, lines_duration_[lines_id_] - new_length));
					//(average_duration - new_length + 1) / 2
					else
						offsets.push_back(0);
				}
			}

			if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW) {
				if (!ReadSegmentFlowToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
											offsets, new_height, new_width, new_length, &datum,
											name_pattern_.c_str())) {
					continue;
				}
			} else if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_RGB_FLOW) {
				if (!ReadSegmentRGBFlowToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
											   offsets, new_height, new_width, new_length, &datum, true,
											   name_pattern_.c_str())) {
					continue;
				}
			} else if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_RGB_POSE) {
//			if (this->phase_==TEST){
//				LOG(INFO) << "Testing video:" << lines_[lines_id_].first;
//			}
				bool roi_pool_flag = this->layer_param_.video_data_param().roi_pool_flag();
				int roi_w = this->layer_param_.video_data_param().roi_w();
				int roi_h = this->layer_param_.video_data_param().roi_h();
//			LOG(INFO) << "LINE ID: " << lines_id_ << ", FNAME: " << lines_[lines_id_].first;

				if (!ReadSegmentRGBPoseToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets,
											   new_height, new_width, new_length, &datum, true, roi_pool_flag,
											   name_pattern_.c_str(), this->person_map, item_id, select_joints,
											   this->layer_param_.video_data_param().stride(), roi_w, roi_h)) {
					continue;
				}

			} else {
				if (!ReadSegmentRGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
										   offsets, new_height, new_width, new_length, &datum, true,
										   name_pattern_.c_str())) {
					continue;
				}
			}

			int offset1 = this->prefetch_data_.offset(item_id);
			this->transformed_data_.set_cpu_data(top_data + offset1);
			this->data_transformer_->Transform(datum, &(this->transformed_data_));
			if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_RGB_POSE &&
				this->layer_param_.video_data_param().roi_part() == VideoDataParameter_ROI_Part_FACE) {
				top_label[item_id] = lines_[lines_id_].second == 0 || lines_[lines_id_].second == 1 ?
									 lines_[lines_id_].second : lines_[lines_id_].second == 77 ? 2 :
																lines_[lines_id_].second == 19 ? 3 : 4;
			} else if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_RGB_POSE &&
					   this->layer_param_.video_data_param().roi_part() == VideoDataParameter_ROI_Part_ARM) {
				if (lines_[lines_id_].second == 98) {
					top_label[item_id] = 0;
				} else if (lines_[lines_id_].second == 22) {
					top_label[item_id] = 1;
				} else if (lines_[lines_id_].second == 23) {
					top_label[item_id] = 2;
				} else if (lines_[lines_id_].second == 32) {
					top_label[item_id] = 3;
				} else if (lines_[lines_id_].second == 34) {
					top_label[item_id] = 4;
				} else if (lines_[lines_id_].second == 55) {
					top_label[item_id] = 5;
				} else if (lines_[lines_id_].second == 57) {
					top_label[item_id] = 6;
				} else if (lines_[lines_id_].second == 78) {
					top_label[item_id] = 7;
				} else {
					top_label[item_id] = 8;
				}
			} else {
				top_label[item_id] = lines_[lines_id_].second;
			}
//		LOG(INFO) << "LABEL: " << top_label[item_id];

			//LOG()

			//next iteration
			if (!this->layer_param_.video_data_param().has_video_id_path()) {
				lines_id_++;
			}

			if (lines_id_ >= lines_size) {
				DLOG(INFO) << "Restarting data prefetching from start.";
				lines_id_ = 0;
				if (this->layer_param_.video_data_param().shuffle()) {
					ShuffleVideos();
				}
			}
		}

	}

	INSTANTIATE_CLASS(VideoDataLayer);

	REGISTER_LAYER_CLASS(VideoData);
}
