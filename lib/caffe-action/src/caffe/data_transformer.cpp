#include <opencv2/core/core.hpp>

#include <string>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/contrib/contrib.hpp>
#include <ctime>
#include <sys/time.h>
#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

	template<typename Dtype>
	DataTransformer<Dtype>::DataTransformer(const TransformationParameter &param,
											Phase phase)
			: param_(param), phase_(phase) {
		// check if we want to use mean_file
		if (param_.has_mean_file()) {
			CHECK_EQ(param_.mean_value_size(), 0) <<
												  "Cannot specify mean_file and mean_value at the same time";
			const string &mean_file = param.mean_file();
			LOG(INFO) << "Loading mean file from: " << mean_file;
			BlobProto blob_proto;
			ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
			data_mean_.FromProto(blob_proto);
		}
		// check if we want to use mean_value
		if (param_.mean_value_size() > 0) {
			CHECK(param_.has_mean_file() == false) <<
												   "Cannot specify mean_file and mean_value at the same time";
			for (int c = 0; c < param_.mean_value_size(); ++c) {
				mean_values_.push_back(param_.mean_value(c));
			}
		}

		//load multiscale info
		max_distort_ = param_.max_distort();
		custom_scale_ratios_.clear();
		for (int i = 0; i < param_.scale_ratios_size(); ++i) {
			custom_scale_ratios_.push_back(param_.scale_ratios(i));
		}
		org_size_proc_ = param.original_image();
	}


/** @build fixed crop offsets for random selection
 */
	void fillFixOffset(int datum_height, int datum_width, int crop_height, int crop_width,
					   bool more_crop,
					   vector<pair<int, int> > &offsets) {
		int height_off = (datum_height - crop_height) / 4;
		int width_off = (datum_width - crop_width) / 4;

		offsets.clear();
		offsets.push_back(pair<int, int>(0, 0)); //upper left
		offsets.push_back(pair<int, int>(0, 4 * width_off)); //upper right
		offsets.push_back(pair<int, int>(4 * height_off, 0)); //lower left
		offsets.push_back(pair<int, int>(4 * height_off, 4 * width_off)); //lower right
		offsets.push_back(pair<int, int>(2 * height_off, 2 * width_off)); //center

		//will be used when more_fix_crop is set to true
		if (more_crop) {
			offsets.push_back(pair<int, int>(0, 2 * width_off)); //top center
			offsets.push_back(pair<int, int>(4 * height_off, 2 * width_off)); //bottom center
			offsets.push_back(pair<int, int>(2 * height_off, 0)); //left center
			offsets.push_back(pair<int, int>(2 * height_off, 4 * width_off)); //right center

			offsets.push_back(pair<int, int>(1 * height_off, 1 * width_off)); //upper left quarter
			offsets.push_back(pair<int, int>(1 * height_off, 3 * width_off)); //upper right quarter
			offsets.push_back(pair<int, int>(3 * height_off, 1 * width_off)); //lower left quarter
			offsets.push_back(pair<int, int>(3 * height_off, 3 * width_off)); //lower right quarter
		}
	}

	float _scale_rates[] = {1.0, .875, .75, .66};
	vector<float> default_scale_rates(_scale_rates, _scale_rates + sizeof(_scale_rates) / sizeof(_scale_rates[0]));

/**
 * @generate crop size when multi-scale cropping is requested
 */
	void fillCropSize(int input_height, int input_width,
					  int net_input_height, int net_input_width,
					  vector<pair<int, int> > &crop_sizes,
					  int max_distort, vector<float> &custom_scale_ratios) {
		crop_sizes.clear();

		vector<float> &scale_rates = (custom_scale_ratios.size() > 0) ? custom_scale_ratios : default_scale_rates;
		int base_size = std::min(input_height, input_width);
		for (int h = 0; h < scale_rates.size(); ++h) {
			int crop_h = int(base_size * scale_rates[h]);
			crop_h = (abs(crop_h - net_input_height) < 3) ? net_input_height : crop_h;
			for (int w = 0; w < scale_rates.size(); ++w) {
				int crop_w = int(base_size * scale_rates[w]);
				crop_w = (abs(crop_w - net_input_width) < 3) ? net_input_width : crop_w;

				//append this cropping size into the list
				if (abs(h - w) <= max_distort) {
					crop_sizes.push_back(pair<int, int>(crop_h, crop_w));
				}
			}
		}
	}

/**
 * @generate crop size and offset when process original images
 */
	void sampleRandomCropSize(int img_height, int img_width,
							  int &crop_height, int &crop_width,
							  float min_scale = 0.08, float max_scale = 1.0, float min_as = 0.75, float max_as = 1.33) {
		float total_area = img_height * img_width;
		float area_ratio = 0;
		float target_area = 0;
		float aspect_ratio = 0;
		float flip_coin = 0;

		int attempt = 0;

		while (attempt < 10) {
			// sample scale and area
			caffe_rng_uniform(1, min_scale, max_scale, &area_ratio);
			target_area = total_area * area_ratio;

			caffe_rng_uniform(1, float(0), float(1), &flip_coin);
			if (flip_coin > 0.5) {
				std::swap(crop_height, crop_width);
			}

			// sample aspect ratio
			caffe_rng_uniform(1, min_as, max_as, &aspect_ratio);
			crop_height = int(sqrt(target_area / aspect_ratio));
			crop_width = int(sqrt(target_area * aspect_ratio));

			if (crop_height <= img_height && crop_width <= img_width) {
				return;
			}
			attempt++;
		}

		// fallback to normal 256-224 style size crop
		crop_height = img_height / 8 * 7;
		crop_width = img_width / 8 * 7;
	}

	template<typename Dtype>
	void DataTransformer<Dtype>::visualize_datum(Dtype *transformed_data, const int width, const int height,
												 const int num_segs,
												 string seg_id, int ch_per_seg) {
		cv::Mat img1(cv::Size(width, height), CV_8UC3), img2(cv::Size(width, height), CV_8UC3),
				flow_x(cv::Size(width, height), CV_8UC1), flow_y(cv::Size(width, height), CV_8UC1);
		for (int c = 0; c < ch_per_seg * num_segs; c++) {
			if (c < 3 * num_segs) {
				for (int h = 0; h < height; h++) {
					for (int w = 0; w < width; w++) {
						int cur_datum_offset = c * height * width + h * width + w;
						int sec_datum_offset = (c + 3 * num_segs) * height * width + h * width + w;
						img1.at<cv::Vec3b>(h, w)[c % 3] = static_cast<uint8_t>(transformed_data[cur_datum_offset]);
						img2.at<cv::Vec3b>(h, w)[c % 3] = static_cast<uint8_t>(transformed_data[sec_datum_offset]);
					}
				}
				if (c % 3 == 2) {
					std::stringstream ss_img1, ss_img2;
					ss_img1 << "visualize/" << seg_id.substr(0, seg_id.npos - 5) << width << "_" << height << "_"
							<< c / 2 << "_img1.jpg";
					ss_img2 << "visualize/" << seg_id.substr(0, seg_id.npos - 5) << width << "_" << height << "_"
							<< c / 2 << "_img2.jpg";

					cv::imwrite(ss_img1.str(), img1);
					cv::imwrite(ss_img2.str(), img2);
					LOG(INFO) << "IMG:" << ss_img1.str() << " WRITTEN";
				}

			} else if (c >= 3 * num_segs && c < 4 * num_segs) {
				for (int h = 0; h < height; h++) {
					for (int w = 0; w < width; w++) {
						int cur_datum_offset = c * height * width + h * width + w;
						int sec_datum_offset = (c + num_segs) * height * width + h * width + w;
						flow_x.at<uchar>(h, w) = static_cast<uint8_t>(transformed_data[cur_datum_offset]);
						flow_y.at<uchar>(h, w) = static_cast<uint8_t>(transformed_data[sec_datum_offset]);
					}
				}
				std::stringstream ss_flowx, ss_flowy;
				ss_flowx << "visualize/" << seg_id.substr(0, seg_id.npos - 5) << width << "_" << height << "_" << c
						 << "_flowx.jpg";
				ss_flowy << "visualize/" << seg_id.substr(0, seg_id.npos - 5) << width << "_" << height << "_" << c
						 << "_flowy.jpg";
				cv::imwrite(ss_flowx.str(), flow_x);
				cv::imwrite(ss_flowy.str(), flow_y);
				LOG(INFO) << "IMG:" << ss_flowx.str() << " WRITTEN";

			}

		}

	}

	void visualize_raw_datum(const string &transformed_data, const int width, const int height,
							 const int num_segs, string seg_id, int ch_per_seg) {
		cv::Mat img1(cv::Size(width, height), CV_8UC3), img2(cv::Size(width, height), CV_8UC3),
				flow_x(cv::Size(width, height), CV_8UC1), flow_y(cv::Size(width, height), CV_8UC1);
		for (int c = 0; c < ch_per_seg * num_segs; c++) {
			if (c < 3 * num_segs) {
				LOG(INFO) << "SEGS:" << num_segs;
				for (int h = 0; h < height; h++) {
					for (int w = 0; w < width; w++) {
						int cur_datum_offset = c * height * width + h * width + w;
						int sec_datum_offset = (c + 3 * num_segs) * height * width + h * width + w;
						img1.at<cv::Vec3b>(h, w)[c % 3] = static_cast<uint8_t>(transformed_data[cur_datum_offset]);
						img2.at<cv::Vec3b>(h, w)[c % 3] = static_cast<uint8_t>(transformed_data[sec_datum_offset]);
					}
				}
				if (c % 3 == 2) {
					std::stringstream ss_img1, ss_img2;
					ss_img1 << "visualize/" << seg_id.substr(0, seg_id.npos - 2) << width << "_" << height << "_"
							<< c / 2 << "_img1.jpg";
					ss_img2 << "visualize/" << seg_id.substr(0, seg_id.npos - 2) << width << "_" << height << "_"
							<< c / 2 << "_img2.jpg";

					cv::imwrite(ss_img1.str(), img1);
					cv::imwrite(ss_img2.str(), img2);
					LOG(INFO) << "IMG:" << ss_img1.str() << " WRITTEN";
				}

			} else if (c >= 3 * num_segs && c < 4 * num_segs) {
				for (int h = 0; h < height; h++) {
					for (int w = 0; w < width; w++) {
						int cur_datum_offset = c * height * width + h * width + w;
						int sec_datum_offset = (c + num_segs) * height * width + h * width + w;
						flow_x.at<uchar>(h, w) = static_cast<uint8_t>(transformed_data[cur_datum_offset]);
						flow_y.at<uchar>(h, w) = static_cast<uint8_t>(transformed_data[sec_datum_offset]);
					}
				}
				std::stringstream ss_flowx, ss_flowy;
				ss_flowx << "visualize/" << seg_id.substr(0, seg_id.npos - 2) << width << "_" << height << "_" << c
						 << "_flowx.jpg";
				ss_flowy << "visualize/" << seg_id.substr(0, seg_id.npos - 2) << width << "_" << height << "_" << c
						 << "_flowy.jpg";
				cv::imwrite(ss_flowx.str(), flow_x);
				cv::imwrite(ss_flowy.str(), flow_y);
				LOG(INFO) << "IMG:" << ss_flowx.str() << " WRITTEN";

			}

		}

	}

	template<typename Dtype>
	void visualize(Dtype *transformed_data, const int width, const int height) {
		cv::Mat img1(cv::Size(width, height), CV_8UC3), img2(cv::Size(width, height), CV_8UC3),
				flow_x(cv::Size(width, height), CV_8UC3), flow_y(cv::Size(width, height), CV_8UC3);
		int img2_offset = 3 * width * height;
		int flow_x_offset = 6 * width * height;
		int flow_y_offset = 7 * width * height;

		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				for (int c = 0; c < 3; c++) {
//					if (c * width * height + h * width + w % 100==0) LOG(INFO) << transformed_data[c * width * height + h * width + w];
					img1.at<cv::Vec3b>(h, w)[c] = (unsigned char) transformed_data[c * width * height + h * width + w];
					img2.at<cv::Vec3b>(h, w)[c] = (unsigned char) transformed_data[img2_offset + c * width * height +
																				   h * width + w];
				}
				flow_x.at<uchar>(h, w) = (unsigned char) transformed_data[flow_x_offset + h * width + w];
				flow_y.at<uchar>(h, w) = (unsigned char) transformed_data[flow_y_offset + h * width + w];
			}
		}
		std::stringstream ss_img1, ss_img2, ss_flowx, ss_flowy;
		ss_img1 << "visualize/" << width << "_" << height << "_" << "img1.jpg";
		ss_img2 << "visualize/" << width << "_" << height << "_" << "img2.jpg";
		ss_flowx << "visualize/" << width << "_" << height << "_" << "flowx.jpg";
		ss_flowy << "visualize/" << width << "_" << height << "_" << "flowy.jpg";

		cv::imwrite(ss_img1.str(), img1);
		cv::imwrite(ss_img2.str(), img2);
		cv::imwrite(ss_flowx.str(), flow_x);
		cv::imwrite(ss_flowy.str(), flow_y);
		LOG(INFO) << "IMG:" << ss_img1.str() << " WRITTEN";

	}

	void visualize_pose_datum(const string &transformed_data, const int width, const int height,
							  const int num_segs, string seg_id) {
		cv::Mat img1(cv::Size(width, height), CV_8UC3), img_pose(cv::Size(width, height), CV_8UC1);
		for (int n_seg = 0; n_seg < num_segs; n_seg++) {
			for (int c = 3 * n_seg; c < 3 * (n_seg + 1); c++) {
				LOG(INFO) << "SEGS:" << num_segs;
				for (int h = 0; h < height; h++) {
					for (int w = 0; w < width; w++) {
						int cur_datum_offset = c * height * width + h * width + w;
						img1.at<cv::Vec3b>(h, w)[c % 3] = static_cast<uint8_t>(transformed_data[cur_datum_offset]);
//						LOG(INFO) << "CUR_PIXEL: "<<static_cast<uint8_t>(transformed_data[cur_datum_offset]);
					}
				}
				if (c % 3 == 2) {
					std::stringstream ss_img1;
					ss_img1 << "visualize/" << seg_id.substr(0, seg_id.npos - 2) << width << "_" << height << "_"
							<< n_seg << "_img1.jpg";
					cv::imwrite(ss_img1.str(), img1);
					LOG(INFO) << "IMG:" << ss_img1.str() << " WRITTEN";
				}
			}
			int c = 3 * num_segs + n_seg;
			for (int h = 0; h < height; h++) {
				for (int w = 0; w < width; w++) {
					int cur_datum_offset = c * height * width + h * width + w;
					img_pose.at<uchar>(h, w) = static_cast<uint8_t>(transformed_data[cur_datum_offset]);
//						LOG(INFO) << "CUR_POSE_PIXEL: "<<static_cast<uint8_t>(transformed_data[cur_datum_offset]);
				}
			}
			std::stringstream ss_flowx;
			ss_flowx << "visualize/" << seg_id.substr(0, seg_id.npos - 2) << width << "_" << height << "_" << n_seg
					 << "_datum_pose.jpg";
			cv::imwrite(ss_flowx.str(), img_pose);

			cv::Mat img_wrighted(cv::Size(width, height), CV_8UC1);
			cv::Mat img_grey;
			cv::cvtColor(img1, img_grey, CV_BGR2GRAY);
			cv::addWeighted(img_grey, 0.5, img_pose, 0.5, 0, img_wrighted);
			std::stringstream ss_pose;
			ss_pose << "visualize/" << seg_id.substr(0, seg_id.npos - 2) << width << "_" << height << "_" << n_seg
					<< "_datum_pose_weighted.jpg";
			cv::imwrite(ss_pose.str(), img_wrighted);
			LOG(INFO) << "IMG:" << ss_pose.str() << " WRITTEN";
		}
	}

	template<typename Dtype>
	void visualize_pose(Dtype *transformed_data, const int width, const int height,
						const int num_segs, string seg_id) {
		LOG(INFO) << "num_segs:" << num_segs;
		LOG(INFO) << "WIDTH:" << width;
		LOG(INFO) << "HEIGHT:" << height;
		cv::Mat img1(cv::Size(width, height), CV_8UC3), img_pose(cv::Size(width, height), CV_8UC1);

		for (int n_seg = 0; n_seg < num_segs; n_seg++) {
			for (int c = 3 * n_seg; c < 3 * (n_seg + 1); c++) {
				LOG(INFO) << "SEGS:" << num_segs;
				for (int h = 0; h < height; h++) {
					for (int w = 0; w < width; w++) {
						int cur_datum_offset = c * height * width + h * width + w;
						img1.at<cv::Vec3b>(h, w)[c % 3] =
								c % 3 == 0 ? static_cast<uint8_t>(transformed_data[cur_datum_offset] + Dtype(104)) :
								c % 3 == 1 ? static_cast<uint8_t>(transformed_data[cur_datum_offset] + Dtype(117)) :
								static_cast<uint8_t>(transformed_data[cur_datum_offset] + Dtype(123));
//						LOG(INFO) << "CUR_PIXEL: "<<static_cast<uint8_t>(transformed_data[cur_datum_offset]);
					}
				}
				if (c % 3 == 2) {
					std::stringstream ss_img1;
					ss_img1 << "visualize/" << seg_id.substr(0, seg_id.npos - 2) << "_trans_pose_" << width << "_" << height
							<< "_" << n_seg << "_img1.jpg";
					cv::imwrite(ss_img1.str(), img1);
					LOG(INFO) << "IMG:" << ss_img1.str() << " WRITTEN";
				}
			}
			int c = 3 * num_segs + n_seg;
			for (int h = 0; h < height; h++) {
				for (int w = 0; w < width; w++) {
					int cur_datum_offset = c * height * width + h * width + w;
					img_pose.at<uchar>(h, w) = static_cast<uint8_t>((transformed_data[cur_datum_offset] - 1.0) * 255);
//						LOG(INFO) << "CUR_POSE_PIXEL: "<<static_cast<uint8_t>(transformed_data[cur_datum_offset]);
				}
			}
			std::stringstream ss_flowx;
			ss_flowx << "visualize/" << seg_id.substr(0, seg_id.npos - 2) << width << "_" << height << "_" << n_seg
					 << "_datum_pose.jpg";
			cv::imwrite(ss_flowx.str(), img_pose);

			cv::Mat img_wrighted(cv::Size(width, height), CV_8UC1);
			cv::Mat img_grey;
			cv::cvtColor(img1, img_grey, CV_BGR2GRAY);
			cv::addWeighted(img_grey, 0.5, img_pose, 0.5, 0, img_wrighted);

			std::stringstream ss_pose;
			ss_pose << "visualize/" << seg_id.substr(0, seg_id.npos - 2) << width << "_" << height << "_" << n_seg
					<< "_datum_pose_weighted.jpg";
			cv::imwrite(ss_pose.str(), img_wrighted);
			LOG(INFO) << "IMG:" << ss_pose.str() << " WRITTEN";
		}


	}


	template<typename Dtype>
	void visualize_roi(Dtype *transformed_data, const int width, const int height, int branch_id,
					   const int num_segs, string seg_id) {
		LOG(INFO) << "num_segs:" << num_segs;
		LOG(INFO) << "WIDTH:" << width;
		LOG(INFO) << "HEIGHT:" << height;
		cv::Mat img1 = cv::Mat::zeros(height, width, CV_8UC3);

		for (int n_seg = branch_id * num_segs; n_seg < (branch_id + 1) * num_segs; n_seg++) {
			for (int c = 3 * n_seg; c < 3 * (n_seg + 1); c++) {
				LOG(INFO) << "SEGS:" << num_segs;
				for (int h = 0; h < height; h++) {
					for (int w = 0; w < width; w++) {
						int cur_datum_offset = c * height * width + h * width + w;
						img1.at<cv::Vec3b>(h, w)[c % 3] =
								c % 3 == 0 ? static_cast<uint8_t>(transformed_data[cur_datum_offset] + Dtype(104)) :
								c % 3 == 1 ? static_cast<uint8_t>(transformed_data[cur_datum_offset] + Dtype(117)) :
								static_cast<uint8_t>(transformed_data[cur_datum_offset] + Dtype(123));
//						LOG(INFO) << "CUR_PIXEL: "<<static_cast<uint8_t>(transformed_data[cur_datum_offset]);
					}
				}
			}
			std::stringstream ss_img1;
			ss_img1 << "visualize/" << seg_id.substr(0, seg_id.npos - 2) << "_trans_roi_" << width << "_" << height
					<< "_" << n_seg << "_img1.jpg";
			cv::imwrite(ss_img1.str(), img1);
			LOG(INFO) << "IMG:" << ss_img1.str() << " WRITTEN";
		}
		img1.release();


	}


	void visualize_datum_roi(const string &transformed_data, const int width, const int height,
							 const int num_segs, string seg_id) {
		LOG(INFO) << "num_segs:" << num_segs;
		LOG(INFO) << "WIDTH:" << width;
		LOG(INFO) << "HEIGHT:" << height;
		cv::Mat img1 = cv::Mat::zeros(height, width, CV_8UC3);

		for (int n_seg = 0; n_seg < num_segs; n_seg++) {
			for (int c = 3 * n_seg; c < 3 * (n_seg + 1); c++) {
				LOG(INFO) << "SEGS:" << num_segs;
				for (int h = 0; h < height; h++) {
					for (int w = 0; w < width; w++) {
						int cur_datum_offset = c * height * width + h * width + w;
						img1.at<cv::Vec3b>(h, w)[c % 3] =
								c % 3 == 0 ? static_cast<uint8_t>(transformed_data[cur_datum_offset] ) :
								c % 3 == 1 ? static_cast<uint8_t>(transformed_data[cur_datum_offset] ) :
								static_cast<uint8_t>(transformed_data[cur_datum_offset] );
//						LOG(INFO) << "CUR_PIXEL: "<<static_cast<uint8_t>(transformed_data[cur_datum_offset]);
					}
				}

			}
			std::stringstream ss_img1;
			ss_img1 << "visualize/" << seg_id.substr(0, seg_id.npos - 2) << "_trans_roi_" << width << "_" << height
					<< "_" << n_seg << "_datum_img1.jpg";
			cv::imwrite(ss_img1.str(), img1);
			LOG(INFO) << "IMG:" << ss_img1.str() << " WRITTEN";

//			int c = 3 * num_segs + n_seg;
//			int roi_offset = 3 * num_segs * height * width + (c - 3 * num_segs) * 5;
//			int top_w = (transformed_data[roi_offset + 1]);
//			int top_h = (transformed_data[roi_offset + 2]);
//			int roi_w = (transformed_data[roi_offset + 3]);
//			int roi_h = (transformed_data[roi_offset + 4]);
//			LOG(INFO) << "roi:(" << top_w << ", " << top_h << ", " << roi_w << ", " << roi_h << ")";
//			cv::Mat img_roi(img1, cv::Rect(top_w, top_h, roi_w, roi_h));
//			std::stringstream ss_flowx;
//			ss_flowx << "visualize/" << seg_id.substr(0, seg_id.npos - 2) << width << "_" << height << "_" << n_seg
//					 << "_datum_roi.jpg";
//			cv::imwrite(ss_flowx.str(), img_roi);

		}
		img1.release();


	}


	template<typename Dtype>
	void DataTransformer<Dtype>::Transform(const Datum &datum,
										   Dtype *transformed_data) {
		timeval tv;
		gettimeofday(&tv, 0);
		long rand_id = tv.tv_usec;

		for (int branch_id = 0; branch_id < param_.is_roi_size(); branch_id++) {

//			LOG(INFO) << param_.is_roi_size() << "branch_id:" << branch_id;
			const string &data = datum.data();
			const int datum_channels = datum.channels();
			const int datum_height = datum.height();
			const int datum_width = datum.width();
			int crop_size = 0;
			int crop_height = 0;
			int crop_width = 0;

			if (param_.has_crop_size())
				crop_size = param_.crop_size();

			if (param_.has_crop_height() && param_.has_crop_width()) {
				crop_height = param_.crop_height();
				crop_width = param_.crop_width();
			}

			const Dtype scale = param_.scale();
			Dtype resize_scale_x = 1.0;
			Dtype resize_scale_y = 1.0;
			bool do_mirror = param_.mirror() && Rand(2);
			const bool has_mean_file = param_.has_mean_file();
			const bool has_uint8 = data.size() > 0;
			const bool has_mean_values = mean_values_.size() > 0;
			const bool do_multi_scale = param_.multi_scale();
			vector<pair<int, int> > offset_pairs;
			vector<pair<int, int> > crop_size_pairs;
			cv::Mat multi_scale_bufferM;



			CHECK_GT(datum_channels, 0);
			if (crop_size) {
				CHECK_GE(datum_height, crop_size);
				CHECK_GE(datum_width, crop_size);
			}
			if (crop_height && crop_width) {
				CHECK_GE(datum_height, crop_height);
				CHECK_GE(datum_width, crop_width);
			}


			if (param_.visualize() && param_.is_rgb_flow()) {
				std::stringstream ss;
				ss << rand_id << "from_raw_datum_";
				visualize_raw_datum(data, datum_width, datum_height, datum_channels / this->param_.num_ch_per_seg(),
									ss.str(), this->param_.num_ch_per_seg());
			} else if (param_.visualize() && param_.is_pose()) {
				std::stringstream ss;
				ss << rand_id;
				visualize_pose_datum(data, datum_width, datum_height, datum_channels / this->param_.num_ch_per_seg(),
									 ss.str());
			} else if (param_.visualize() && param_.is_roi(branch_id)) {
				std::stringstream ss;
				ss << rand_id;
				visualize_datum_roi(data, datum_width, datum_height, datum_channels / this->param_.num_ch_per_seg(),
									ss.str());
			}

			Dtype *mean = NULL;
			if (has_mean_file) {
				CHECK_EQ(datum_channels, data_mean_.channels());
				CHECK_EQ(datum_height, data_mean_.height());
				CHECK_EQ(datum_width, data_mean_.width());
				mean = data_mean_.mutable_cpu_data();
			}
			if (has_mean_values) {
				CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels)
				<< "Specify either 1 mean_value or as many as channels: " << datum_channels;
				if (datum_channels > 1 && mean_values_.size() == 1) {
					// Replicate the mean_value for simplicity
					for (int c = 1; c < datum_channels; ++c) {
						mean_values_.push_back(mean_values_[0]);
					}
				}
			}

			if (!(crop_size || (crop_height && crop_width)) && do_multi_scale) {
				LOG(ERROR) << "Multi scale augmentation is only activated with crop_size set.";
			}


			int height = datum_height;
			int width = datum_width;
			int rescale_height = 0;
			int rescale_width = 0;
			int h_off = 0;
			int w_off = 0;
			bool need_imgproc = true;
			bool is_roi = param_.is_roi(branch_id);

			if (crop_size || (crop_height && crop_width)) {
//			height = crop_size;
//			width = crop_size;
				height = crop_height;
				width = crop_width;
				// We only do random crop when we do training.
				if (phase_ == TRAIN) {
					// If in training and we need multi-scale cropping, reset the crop size params
					if (do_multi_scale) {
						fillCropSize(datum_height, datum_width, crop_height, crop_width, crop_size_pairs,
									 max_distort_, custom_scale_ratios_);
						int sel = Rand(crop_size_pairs.size()); // select a scale (determine the width & height)
						rescale_height = crop_size_pairs[sel].first;
						rescale_width = crop_size_pairs[sel].second;
//					LOG(INFO) << "crop_HEIGHT: " << crop_height;
//					LOG(INFO) << "crop_WIDTH: " << crop_width;
//					LOG(INFO) << "HEIGHT: " << rescale_height;
//					LOG(INFO) << "WIDTH: " << rescale_width;
						if (param_.do_rescale()) {
							resize_scale_x = Dtype(crop_width) / Dtype(rescale_width);
							resize_scale_y = Dtype(crop_height) / Dtype(rescale_height);
						}

					} else {
						rescale_height = crop_height;
						rescale_width = crop_width;
					}
					if (param_.fix_crop()) {
						fillFixOffset(datum_height, datum_width, rescale_height, rescale_width,
									  param_.more_fix_crop(), offset_pairs);
						int sel = Rand(offset_pairs.size());
						h_off = offset_pairs[sel].first;
						w_off = offset_pairs[sel].second;
					} else {
						//TODO: ADD RANDOM SCALE HERE
						h_off = Rand(datum_height - rescale_height + 1);
						w_off = Rand(datum_width - rescale_width + 1);
					}

				} else {
					rescale_height = crop_height;
					rescale_width = crop_width;
					if (param_.fix_crop() && (!is_roi)) {
						fillFixOffset(datum_height, datum_width, rescale_height, rescale_width,
									  param_.more_fix_crop(), offset_pairs);
						int sel = 0;
						if (param_.has_oversample_type_path() && phase_ == TEST) {
							string oversample_type_path = param_.oversample_type_path();
							std::ifstream foversample_type(oversample_type_path.c_str());
							string oversample_type_str;

							getline(foversample_type, oversample_type_str);

							sel = std::atoi(oversample_type_str.c_str());
//						LOG(INFO) << "OVERSAMPLE_ID:" << oversample_type_str << "SEL: " << sel;
							if (sel >= offset_pairs.size()) { // do_mirror
								sel -= offset_pairs.size();
								do_mirror = true;
							} else {
								do_mirror = false;
							}
							foversample_type.close();
						} else {
							sel = Rand(offset_pairs.size());
						}

						h_off = offset_pairs[sel].first;
						w_off = offset_pairs[sel].second;
					} else {
//					rescale_height = crop_height;
//					rescale_width = crop_width;
						h_off = (datum_height - crop_height) / 2;
						w_off = (datum_width - crop_width) / 2;
					}

				}
			}
			need_imgproc = true;//do_multi_scale && (crop_size || (crop_height && crop_width));
//					   && ((crop_height != rescale_height) || (crop_width != rescale_width));

			Dtype datum_element;
			int top_index, data_index;

			const int num_segs = datum_channels / this->param_.num_ch_per_seg();

			float select_mult;
			int translation_h;
			int translation_w;
			if (is_roi) {
				if (param_.has_select_mult_path()) {
					string select_mult_path = param_.select_mult_path();
					std::ifstream fscale_mult(select_mult_path.c_str());
					string select_mult_str;
					getline(fscale_mult, select_mult_str);
					select_mult = std::strtof(select_mult_str.c_str(), NULL);
					fscale_mult.close();
				} else {
					select_mult =
							phase_ == TRAIN ? param_.translation_mult(rand() % param_.translation_mult_size()) : 2.5;
				}
				translation_h = param_.translation_bound() == 0 ? 0 : rand() % param_.translation_bound();
				translation_w = param_.translation_bound() == 0 ? 0 : rand() % param_.translation_bound();
			}

			for (int c = 0; c < datum_channels; ++c) {
				// first_img slice: [0, 3 * num_segs] channels
				// second_img slice: [3 * num_segs, 6 * num_segs] channels
				// flow_x slice: [6 * num_segs, 7 * num_segs] channels
				// flow_y slice: [7 * num_segs, 8 * num_segs] channels
				// datum_channels = 8 * num_segs
				bool is_rgb_flow = param_.is_rgb_flow() && c >= num_segs * 6;
				bool is_flow = param_.is_flow() ;
				bool is_pose = param_.is_pose() && c >= num_segs * 3;
				//&& c >= num_segs * 3
//			LOG(INFO) << "IS_POSE: " <<is_pose<<", c: " << c << "num_segs: " << num_segs;

				// image resize etc needed

				if (is_roi) {

					float scale_h = float(datum_height) / float(datum_height);
					float scale_w = float(datum_width) / float(datum_width);

//				LOG(INFO) << "scale_h:" << scale_h << ", scale_w:" << scale_w;
//				data_index = (3 * num_segs * datum_height) * datum_width + (c - 3 * num_segs) * 5;
					int coor_index = (c / 3) * 5;
//					top_index = 3 * num_segs * height * width + (c - 3 * num_segs) * 5;
					int datum_top_w = coor_vec[coor_index + 1] + translation_w;
					int datum_top_h = coor_vec[coor_index + 2] + translation_h;
					int datum_roi_w = coor_vec[coor_index + 3];
					int datum_roi_h = coor_vec[coor_index + 4];
//				LOG(INFO) << "datum_top_w: " << datum_top_w << "datum_top_h: "<<datum_top_h << "datum_roi_w: "<<datum_roi_w
//				<< "datum_roi_h: "<<datum_roi_h;
//				roi_index += 1.0;
					//calculate intersection rect
					cv::Point2f intersect_top(std::max(datum_top_w, 0), std::max(datum_top_h, 0));
					cv::Point2f intersect_bot(std::min(datum_top_w + datum_roi_w, datum_width),
											  std::min(datum_top_h + datum_roi_h, datum_height));

					Dtype roi_top_w, roi_top_h, roi_w, roi_h;
					if (!param_.has_select_mult_path()) {
						select_mult = phase_ == TRAIN ?
									  c % 3 == 0 ? param_.translation_mult(rand() % param_.translation_mult_size()) :
									  select_mult : 2.5;
					}
					int trans_mult = rand() % 2 == 1 ? 1 : -1;
					translation_h = phase_ == TRAIN && c % 3 == 0 ? param_.translation_bound() != 0 ?
																	(rand() % param_.translation_bound()) * trans_mult
																									: 0 : 0;
					translation_w = phase_ == TRAIN && c % 3 == 0 ? param_.translation_bound() != 0 ?
																	(rand() % param_.translation_bound()) * trans_mult
																									: 0 : 0;
					if (intersect_top.x - intersect_bot.x <= 0 && intersect_top.y - intersect_bot.y <= 0) { // intersect

						roi_top_w = (intersect_top.x) * scale_w - (datum_roi_w * select_mult - datum_roi_w) / 2 > 0 ?
									(intersect_top.x) * scale_w - (datum_roi_w * select_mult - datum_roi_w) / 2 : 0;
						roi_top_h = (intersect_top.y) * scale_h - (datum_roi_h * select_mult - datum_roi_h) / 2 > 0 ?
									(intersect_top.y) * scale_h - (datum_roi_h * select_mult - datum_roi_h) / 2 : 0;
						roi_w = (intersect_bot.x - intersect_top.x) * scale_w * select_mult > datum_width - roi_top_w ?
								datum_width - roi_top_w : (intersect_bot.x - intersect_top.x) * scale_w * select_mult;
						roi_h = (intersect_bot.y - intersect_top.y) * scale_h * select_mult > datum_height - roi_top_h ?
								datum_height - roi_top_h : (intersect_bot.y - intersect_top.y) * scale_h * select_mult;
					} else {
						roi_top_w = 0;
						roi_top_h = 0;
						roi_w = 0;
						roi_h = 0;
					}


					cv::Mat M(datum_height, datum_width,
							  has_uint8 ? CV_8UC1 : CV_32FC1); // TODO: M is the temp Mat for each channel in datum

					//put the datum content to a cvMat
					for (int h = 0; h < datum_height; ++h) {
						for (int w = 0; w < datum_width; ++w) {
							int data_index = (c * datum_height + h) * datum_width + w;
							if (has_uint8) {
								M.at<uchar>(h, w) = static_cast<uint8_t>(data[data_index]);
							} else {
								M.at<float>(h, w) = datum.float_data(data_index);
							}
						}
					}

					//resize the cropped patch to network input size
					stringstream ss_c;
					if (param_.is_roi(branch_id) && param_.visualize())
						ss_c << "visualize/" << rand_id << "_img_channel_" << c;
//					cv::copyMakeBorder(M, padM, border_h, border_h, border_w, border_w, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
					if (roi_w > 10 && roi_h > 10) {
						if (param_.is_roi(branch_id) && param_.visualize()) ss_c << "_roi_crop.jpg";
						cv::Mat cropM(M, cv::Rect(int(roi_top_w), int(roi_top_h), (int) roi_w, (int) roi_h));
						cv::resize(cropM, multi_scale_bufferM, cv::Size(crop_width, crop_height));
						cropM.release();
					} else if (param_.use_zero()) {
						multi_scale_bufferM = cv::Mat::zeros(crop_height, crop_width, has_uint8 ? CV_8UC1 : CV_32FC1);
					} else {
						if (param_.is_roi(branch_id) && param_.visualize()) ss_c << "_norm_crop.jpg";
						cv::Mat cropM(M, cv::Rect(w_off, h_off, rescale_width, rescale_height));
						cv::resize(cropM, multi_scale_bufferM, cv::Size(crop_width, crop_height));

						cropM.release();
					}
					if (param_.is_roi(branch_id) && param_.visualize()) cv::imwrite(ss_c.str(), multi_scale_bufferM);
					if (param_.is_roi(branch_id) && param_.visualize()) LOG(INFO) << ss_c.str() << " WRITTEN!";
					M.release();

//				LOG(INFO) << "id:" << roi_index << ", roi_top_w: " << roi_top_w << ", roi_top_h: "
//						  << roi_top_h << ", roi_w: " << roi_w << ", roi_h: " << roi_h;
				} else {
					cv::Mat M(datum_height, datum_width,
							  has_uint8 ? CV_8UC1 : CV_32FC1); // TODO: M is the temp Mat for each channel in datum
//					LOG(INFO) << "datum_height:" << datum_height << ", datum_width: " << datum_width;
					//put the datum content to a cvMat
					for (int h = 0; h < datum_height; ++h) {
						for (int w = 0; w < datum_width; ++w) {
							int data_index = (c * datum_height + h) * datum_width + w;
							if (has_uint8) {
								M.at<uchar>(h, w) = static_cast<uint8_t>(data[data_index]);
							} else {
								M.at<float>(h, w) = datum.float_data(data_index);
							}
						}
					}
					//resize the cropped patch to network input size
//				LOG(INFO) << "w_off: " << w_off;
//				LOG(INFO) << "h_off: " << h_off;
//				LOG(INFO) << "rescale_width: " << rescale_width;
//				LOG(INFO) << "rescale_height: " << rescale_height;
//				LOG(INFO) << "datum_height: " << datum_height;
//				LOG(INFO) << "datum_width: " << datum_width;
					cv::Mat cropM(M, cv::Rect(w_off, h_off, rescale_width, rescale_height));
//				LOG(INFO) << "w_off" << w_off;
//				LOG(INFO) << "h_off" << h_off;
//				LOG(INFO) << "rescale_width" << rescale_width;
//				LOG(INFO) << "rescale_height" << rescale_height;
//				cv::resize(cropM, multi_scale_bufferM, cv::Size(crop_size, crop_size));
					cv::resize(cropM, multi_scale_bufferM, cv::Size(crop_width, crop_height));
					M.release();
					cropM.release();
				}
//				LOG(INFO) << "1";

				for (int h = 0; h < height; ++h) {
					for (int w = 0; w < width; ++w) {
						data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
						if (do_mirror) {
							top_index = (branch_id * datum_channels * height * width) +
										(c * height + h) * width + (width - 1 - w);
						} else {
							top_index = (branch_id * datum_channels * height * width) +
										(c * height + h) * width + w;
						}
						if (need_imgproc) {
							if (has_uint8) {
								if (is_flow && do_mirror&& c%2 == 0)
									datum_element = 255 - static_cast<Dtype>(multi_scale_bufferM.at<uint8_t>(h, w));
								else if (is_pose)
									datum_element =
											static_cast<Dtype>(multi_scale_bufferM.at<uint8_t>(h, w)) / 255.0 + 1.0;
								else
									datum_element = static_cast<Dtype>(multi_scale_bufferM.at<uint8_t>(h, w));
							} else {
								if (is_flow && do_mirror&& c%2 == 0)
									datum_element = 255 - static_cast<Dtype>(multi_scale_bufferM.at<float>(h, w));
								else if (is_pose)
									datum_element =
											static_cast<Dtype>(multi_scale_bufferM.at<float>(h, w)) / 255.0 + 1.0;
								else
									datum_element = static_cast<Dtype>(multi_scale_bufferM.at<float>(h, w));
							}
						} else {
							if (has_uint8) {
								if (is_flow && do_mirror&& c%2 == 0)
									datum_element = 255 - static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
								else if (is_pose)
									datum_element =
											static_cast<Dtype>(static_cast<uint8_t>(data[data_index])) / 255.0 + 1.0;
								else
									datum_element = static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
							} else {
								if (is_flow && do_mirror&& c%2 == 0)
									datum_element = 255 - datum.float_data(data_index);
								else if (is_pose)
									datum_element = datum.float_data(data_index) / 255.0 + 1.0;
								else
									datum_element = datum.float_data(data_index);
							}
						}
#define RESCALE_FLOW(v, s) (s * v - 255 * (s - 1) / 2)
						if (has_mean_file) {
							if (do_multi_scale) {
								int fixed_data_index = (c * datum_height + h) * datum_width + w;
								if (is_rgb_flow) {
									transformed_data[top_index] = c >= 7 * num_segs ?
																  RESCALE_FLOW(datum_element, resize_scale_y) -
																  mean[fixed_data_index] :
																  RESCALE_FLOW(datum_element, resize_scale_x) -
																  mean[fixed_data_index];
								} else {
									transformed_data[top_index] = datum_element - mean[fixed_data_index];
								}
							} else {
								if (is_rgb_flow) {
									transformed_data[top_index] = c >= 7 * num_segs ?
																  (RESCALE_FLOW(datum_element, resize_scale_y) -
																   mean[data_index]) * scale :
																  (RESCALE_FLOW(datum_element, resize_scale_x) -
																   mean[data_index]) * scale;

								} else {
									transformed_data[top_index] = (datum_element - mean[data_index]) * scale;
								}
							}
						} else {
							if (has_mean_values) {
								if (is_rgb_flow) {
									transformed_data[top_index] = c >= 7 * num_segs ?
																  (RESCALE_FLOW(datum_element, resize_scale_y) -
																   mean_values_[c]) * scale :
																  (RESCALE_FLOW(datum_element, resize_scale_x) -
																   mean_values_[c]) * scale;
								} else {
									transformed_data[top_index] = (datum_element - mean_values_[c]) * scale;
								}
							} else {
								if (is_rgb_flow) {
									transformed_data[top_index] = c >= 7 * num_segs ?
																  RESCALE_FLOW(datum_element, resize_scale_y) * scale :
																  RESCALE_FLOW(datum_element, resize_scale_x) * scale;
								} else {
									transformed_data[top_index] = datum_element * scale;
								}

							}
						}
#undef RESCALE_FLOW
					}
				}
//				LOG(INFO) << c << " loop done!";

			}
			if (param_.visualize() && param_.is_rgb_flow()) {
				std::stringstream ss;
				ss << rand_id;
				visualize_datum(transformed_data, width, height, num_segs, ss.str(), this->param_.num_ch_per_seg());
			} else if (param_.is_roi(branch_id) && param_.visualize()) {
				std::stringstream ss;
				ss << rand_id;
				visualize_roi(transformed_data, width, height, branch_id, num_segs, ss.str());
			} else if (param_.visualize()) {
				std::stringstream ss;
				ss << rand_id << "rgb_";
				visualize_roi(transformed_data, width, height, branch_id, num_segs, ss.str());
			}
			multi_scale_bufferM.release();
		}

	}


	template<typename Dtype>
	void DataTransformer<Dtype>::Transform(const Datum &datum_data, const Datum &datum_label,
										   Blob<Dtype> *transformed_data, Blob<Dtype> *transformed_label) {


		CHECK_EQ(datum_data.height(), datum_label.height());
		CHECK_EQ(datum_data.width(), datum_label.width());

		const string &data = datum_data.data();
		const string &label = datum_label.data();
		const int datum_channels = datum_data.channels();
		const int datum_height = datum_data.height();
		const int datum_width = datum_data.width();

		float lower_scale = 1, upper_scale = 1;
		if (param_.scale_ratios_size() == 2) {
			lower_scale = param_.scale_ratios(0);
			upper_scale = param_.scale_ratios(1);
		}
		const Dtype scale = param_.scale();
		const bool do_mirror = param_.mirror() && Rand(2);
		const bool has_mean_file = param_.has_mean_file();
		const bool has_mean_values = mean_values_.size() > 0;
		const int stride = param_.stride();

		CHECK_GT(datum_channels, 0);

		if (has_mean_file) {
			NOT_IMPLEMENTED;
		}
		if (has_mean_values) {
			CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
																					 "Specify either 1 mean_value or as many as channels: "
																					 << datum_channels;
			if (datum_channels > 1 && mean_values_.size() == 1) {
				// Replicate the mean_value for simplicity
				for (int c = 1; c < datum_channels; ++c) {
					mean_values_.push_back(mean_values_[0]);
				}
			}
		}

		float scale_ratios = std::max(Rand(int((upper_scale - lower_scale) * 1000.0) + 1) / 1000.0, 0.0) + lower_scale;

		int height = int(datum_height * scale_ratios + 0.5);
		int width = int(datum_width * scale_ratios + 0.5);


		int crop_height = height / stride * stride;
		int crop_width = width / stride * stride;

		if (param_.has_upper_size()) {
			crop_height = std::min(crop_height, param_.upper_size());
			crop_width = std::min(crop_width, param_.upper_size());
		} else if (param_.has_upper_height() && param_.has_upper_width()) {
			crop_height = std::min(crop_height, param_.upper_height());
			crop_width = std::min(crop_width, param_.upper_width());
		}


		int h_off = Rand(height - crop_height + 1);
		int w_off = Rand(width - crop_width + 1);

		transformed_data->Reshape(1, datum_channels, crop_height, crop_width);
		transformed_label->Reshape(1, 1, crop_height, crop_width);

		//for image data

		Dtype datum_element;
		int top_index;
		Dtype *ptr = transformed_data->mutable_cpu_data();
		for (int c = 0; c < datum_channels; ++c) {
			cv::Mat M(datum_height, datum_width, CV_8UC1);
			for (int h = 0; h < datum_height; ++h)
				for (int w = 0; w < datum_width; ++w) {
					int data_index = (c * datum_height + h) * datum_width + w;
					M.at<uchar>(h, w) = static_cast<uint8_t>(data[data_index]);
				}
			cv::resize(M, M, cv::Size(width, height));
			cv::Mat cropM(M, cv::Rect(w_off, h_off, crop_width, crop_height));
			for (int h = 0; h < crop_height; ++h)
				for (int w = 0; w < crop_width; ++w) {

					if (do_mirror)
						top_index = (c * crop_height + h) * crop_width + (crop_width - 1 - w);
					else
						top_index = (c * crop_height + h) * crop_width + w;

					datum_element = static_cast<Dtype>(cropM.at<uint8_t>(h, w));
					if (has_mean_file) {
						NOT_IMPLEMENTED;
					} else if (has_mean_values)
						ptr[top_index] = (datum_element - mean_values_[c]) * scale;
					else
						ptr[top_index] = datum_element * scale;
				}
			M.release();
			cropM.release();
		}

		//for label

		ptr = transformed_label->mutable_cpu_data();
		cv::Mat M(datum_height, datum_width, CV_8UC1);
		for (int h = 0; h < datum_height; ++h)
			for (int w = 0; w < datum_width; ++w) {
				int data_index = h * datum_width + w;
				M.at<uchar>(h, w) = static_cast<uint8_t>(label[data_index]);
			}
		cv::resize(M, M, cv::Size(width, height), 0, 0, CV_INTER_NN);
		cv::Mat cropM(M, cv::Rect(w_off, h_off, crop_width, crop_height));
		for (int h = 0; h < crop_height; ++h)
			for (int w = 0; w < crop_width; ++w) {

				if (do_mirror)
					top_index = h * crop_width + (crop_width - 1 - w);
				else
					top_index = h * crop_width + w;

				ptr[top_index] = static_cast<Dtype>(cropM.at<uint8_t>(h, w));
			}
		M.release();
		cropM.release();
	}


	template<typename Dtype>
	void DataTransformer<Dtype>::Transform(const Datum &datum,
										   Blob<Dtype> *transformed_blob) {
		// If datum is encoded, decoded and transform the cv::image.
		if (datum.encoded()) {
			CHECK(!(param_.force_color() && param_.force_gray()))
			<< "cannot set both force_color and force_gray";
			cv::Mat cv_img;
			if (param_.force_color() || param_.force_gray()) {
				// If force_color then decode in color otherwise decode in gray.
				cv_img = DecodeDatumToCVMat(datum, param_.force_color());
			} else {
				cv_img = DecodeDatumToCVMatNative(datum);
			}
			// Transform the cv::image into blob.
			return Transform(cv_img, transformed_blob);
		} else {
			if (param_.force_color() || param_.force_gray()) {
				LOG(ERROR) << "force_color and force_gray only for encoded datum";
			}
		}

		int crop_size;
		int crop_height;
		int crop_width;
		if (param_.has_crop_size())
			crop_size = param_.crop_size();
		if (param_.has_crop_height() && param_.has_crop_width()) {
			crop_height = param_.crop_height();
			crop_width = param_.crop_width();
		}
//		LOG(INFO) << "crop_height: "<< crop_height;
//		LOG(INFO) << "crop_width: " << crop_width;
//		LOG(INFO) << "crop_size:" << crop_size;

		const int datum_channels = datum.channels();
		const int datum_height = datum.height();
		const int datum_width = datum.width();

		// Check dimensions.
		const int channels = transformed_blob->channels();
		const int height = transformed_blob->height();
		const int width = transformed_blob->width();
		const int num = transformed_blob->num();

//		CHECK_EQ(channels, datum_channels);
		CHECK_LE(height, datum_height);
		CHECK_LE(width, datum_width);
		CHECK_GE(num, 1);

		if (crop_height && crop_width) {
//			CHECK_EQ(crop_height, height);
//			CHECK_EQ(crop_width, width);
		} else if (crop_size) {
			CHECK_EQ(crop_size, height);
			CHECK_EQ(crop_size, width);
		} else {
			CHECK_EQ(datum_height, height);
			CHECK_EQ(datum_width, width);
		}

		Dtype *transformed_data = transformed_blob->mutable_cpu_data();
		Transform(datum, transformed_data);
	}

	template<typename Dtype>
	void DataTransformer<Dtype>::Transform(const vector<Datum> &datum_vector,
										   Blob<Dtype> *transformed_blob) {
		const int datum_num = datum_vector.size();
		const int num = transformed_blob->num();
		const int channels = transformed_blob->channels();
		const int height = transformed_blob->height();
		const int width = transformed_blob->width();

		CHECK_GT(datum_num, 0) << "There is no datum to add";
		CHECK_LE(datum_num, num) <<
								 "The size of datum_vector must be no greater than transformed_blob->num()";
		Blob<Dtype> uni_blob(1, channels, height, width);
		for (int item_id = 0; item_id < datum_num; ++item_id) {
			int offset = transformed_blob->offset(item_id);
			uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
			Transform(datum_vector[item_id], &uni_blob);
		}
	}

	template<typename Dtype>
	void DataTransformer<Dtype>::Transform(const vector<cv::Mat> &mat_vector,
										   Blob<Dtype> *transformed_blob) {
		const int mat_num = mat_vector.size();
		const int num = transformed_blob->num();
		const int channels = transformed_blob->channels();
		const int height = transformed_blob->height();
		const int width = transformed_blob->width();

		CHECK_GT(mat_num, 0) << "There is no MAT to add";
		CHECK_EQ(mat_num, num) <<
							   "The size of mat_vector must be equals to transformed_blob->num()";
		Blob<Dtype> uni_blob(1, channels, height, width);
		for (int item_id = 0; item_id < mat_num; ++item_id) {
			int offset = transformed_blob->offset(item_id);
			uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
			Transform(mat_vector[item_id], &uni_blob);
		}
	}

	template<typename Dtype>
	void DataTransformer<Dtype>::Transform(const cv::Mat &cv_img,
										   Blob<Dtype> *transformed_blob) {
		const int crop_size = param_.crop_size();
		const int img_channels = cv_img.channels();
		const int img_height = cv_img.rows;
		const int img_width = cv_img.cols;

		// Check dimensions.
		const int channels = transformed_blob->channels();
		const int height = transformed_blob->height();
		const int width = transformed_blob->width();
		const int num = transformed_blob->num();

		CHECK_EQ(channels, img_channels);

		if (!org_size_proc_) {
			CHECK_LE(height, img_height);
			CHECK_LE(width, img_width);
		}
		CHECK_GE(num, 1);

		CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

		const Dtype scale = param_.scale();
		const bool do_mirror = param_.mirror() && Rand(2);
		const bool has_mean_file = param_.has_mean_file();
		const bool has_mean_values = mean_values_.size() > 0;
		const bool do_multi_scale = param_.multi_scale();

		vector<pair<int, int> > offset_pairs;
		vector<pair<int, int> > crop_size_pairs;

		cv::Mat cv_cropped_img;

		CHECK_GT(img_channels, 0);
		if (!org_size_proc_) {
			CHECK_GE(img_height, crop_size);
			CHECK_GE(img_width, crop_size);
		}

		Dtype *mean = NULL;
		if (has_mean_file) {
			CHECK_EQ(img_channels, data_mean_.channels());
			CHECK_EQ(img_height, data_mean_.height());
			CHECK_EQ(img_width, data_mean_.width());
			mean = data_mean_.mutable_cpu_data();
		}
		if (has_mean_values) {
			CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
																				   "Specify either 1 mean_value or as many as channels: "
																				   <<
																				   img_channels;
			if (img_channels > 1 && mean_values_.size() == 1) {
				// Replicate the mean_value for simplicity
				for (int c = 1; c < img_channels; ++c) {
					mean_values_.push_back(mean_values_[0]);
				}
			}
		}

		int h_off = 0;
		int w_off = 0;
		int crop_height = 0;
		int crop_width = 0;

		if (!org_size_proc_) {
			if (crop_size) {
				CHECK_EQ(crop_size, height);
				CHECK_EQ(crop_size, width);
				// We only do random crop when we do training.
				if (phase_ == TRAIN) {
					if (do_multi_scale) {
						fillCropSize(img_height, img_width, crop_size, crop_size, crop_size_pairs,
									 max_distort_, custom_scale_ratios_);
						int sel = Rand(crop_size_pairs.size());
						crop_height = crop_size_pairs[sel].first;
						crop_width = crop_size_pairs[sel].second;
					} else {
						crop_height = crop_size;
						crop_width = crop_size;
					}
					if (param_.fix_crop()) {
						fillFixOffset(img_height, img_width, crop_height, crop_width,
									  param_.more_fix_crop(), offset_pairs);
						int sel = Rand(offset_pairs.size());
						h_off = offset_pairs[sel].first;
						w_off = offset_pairs[sel].second;
					} else {
						h_off = Rand(img_height - crop_height + 1);
						w_off = Rand(img_width - crop_width + 1);
					}
				} else {
					h_off = (img_height - crop_size) / 2;
					w_off = (img_width - crop_size) / 2;
					crop_width = crop_size;
					crop_height = crop_size;
				}
				cv::Rect roi(w_off, h_off, crop_width, crop_height);
				// if resize needed, first put the resized image into a buffer, then copy back.
				if (do_multi_scale && ((crop_height != crop_size) || (crop_width != crop_size))) {
					cv::Mat crop_bufferM(cv_img, roi);
					cv::resize(crop_bufferM, cv_cropped_img, cv::Size(crop_size, crop_size));
					crop_bufferM.release();
				} else {
					cv_cropped_img = cv_img(roi);
				}
			} else {
				CHECK_EQ(img_height, height);
				CHECK_EQ(img_width, width);
				cv_cropped_img = cv_img;
			}
		} else {
			CHECK(crop_size > 0) << "in original image processing mode, crop size must be specified";
			CHECK_EQ(crop_size, height);
			CHECK_EQ(crop_size, width);
			if (phase_ == TRAIN) {
				// in training, we randomly crop different sized crops
				sampleRandomCropSize(img_height, img_width, crop_height, crop_width);


				h_off = (crop_height < img_height) ? Rand(img_height - crop_height) : 0;
				w_off = (crop_width < img_width) ? Rand(img_width - crop_width) : 0;
			} else {
				// in testing, we first resize the image to sizeof (8/7*crop_size) then crop the central patch
				h_off = img_height / 14;
				w_off = img_width / 14;
				crop_height = img_height / 8 * 7;
				crop_width = img_width / 8 * 7;
			}

			cv::Rect roi(w_off, h_off, crop_width, crop_height);

			// resize is always needed in original image mode
			cv::Mat crop_bufferM(cv_img, roi);
			cv::resize(crop_bufferM, cv_cropped_img, cv::Size(crop_size, crop_size), 0, 0, CV_INTER_CUBIC);
			crop_bufferM.release();
		}

		CHECK(cv_cropped_img.data);

		Dtype *transformed_data = transformed_blob->mutable_cpu_data();
		int top_index;
		for (int h = 0; h < height; ++h) {
			const uchar *ptr = cv_cropped_img.ptr<uchar>(h);
			int img_index = 0;
			for (int w = 0; w < width; ++w) {
				for (int c = 0; c < img_channels; ++c) {
					if (do_mirror) {
						top_index = (c * height + h) * width + (width - 1 - w);
					} else {
						top_index = (c * height + h) * width + w;
					}
					// int top_index = (c * height + h) * width + w;
					Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
					if (has_mean_file) {
						//we will use a fixed position of mean map for multi-scale.
						int mean_index = (do_multi_scale) ?
										 (c * img_height + h) * img_width + w
														  : (c * img_height + h_off + h) * img_width + w_off + w;
						if (param_.is_flow() && do_mirror && c % 2 == 0)
							transformed_data[top_index] =
									(255 - pixel - mean[mean_index]) * scale;
						else
							transformed_data[top_index] =
									(pixel - mean[mean_index]) * scale;
					} else {
						if (has_mean_values) {
							if (param_.is_flow() && do_mirror && c % 2 == 0)
								transformed_data[top_index] =
										(255 - pixel - mean_values_[c]) * scale;
							else
								transformed_data[top_index] =
										(pixel - mean_values_[c]) * scale;
						} else {
							if (param_.is_flow() && do_mirror && c % 2 == 0)
								transformed_data[top_index] = (255 - pixel) * scale;
							else
								transformed_data[top_index] = pixel * scale;
						}
					}
				}
			}
		}
		cv_cropped_img.release();
	}

	template<typename Dtype>
	void DataTransformer<Dtype>::Transform(Blob<Dtype> *input_blob,
										   Blob<Dtype> *transformed_blob) {
		const int crop_size = param_.crop_size();
		const int input_num = input_blob->num();
		const int input_channels = input_blob->channels();
		const int input_height = input_blob->height();
		const int input_width = input_blob->width();

		if (transformed_blob->count() == 0) {
			// Initialize transformed_blob with the right shape.
			if (crop_size) {
				transformed_blob->Reshape(input_num, input_channels,
										  crop_size, crop_size);
			} else {
				transformed_blob->Reshape(input_num, input_channels,
										  input_height, input_width);
			}
		}

		const int num = transformed_blob->num();
		const int channels = transformed_blob->channels();
		const int height = transformed_blob->height();
		const int width = transformed_blob->width();
		const int size = transformed_blob->count();

		CHECK_LE(input_num, num);
		CHECK_EQ(input_channels, channels);
		CHECK_GE(input_height, height);
		CHECK_GE(input_width, width);


		const Dtype scale = param_.scale();
		const bool do_mirror = param_.mirror() && Rand(2);
		const bool has_mean_file = param_.has_mean_file();
		const bool has_mean_values = mean_values_.size() > 0;

		int h_off = 0;
		int w_off = 0;
		if (crop_size) {
			CHECK_EQ(crop_size, height);
			CHECK_EQ(crop_size, width);
			// We only do random crop when we do training.
			if (phase_ == TRAIN) {
				h_off = Rand(input_height - crop_size + 1);
				w_off = Rand(input_width - crop_size + 1);
			} else {
				h_off = (input_height - crop_size) / 2;
				w_off = (input_width - crop_size) / 2;
			}
		} else {
			CHECK_EQ(input_height, height);
			CHECK_EQ(input_width, width);
		}

		Dtype *input_data = input_blob->mutable_cpu_data();
		if (has_mean_file) {
			CHECK_EQ(input_channels, data_mean_.channels());
			CHECK_EQ(input_height, data_mean_.height());
			CHECK_EQ(input_width, data_mean_.width());
			for (int n = 0; n < input_num; ++n) {
				int offset = input_blob->offset(n);
				caffe_sub(data_mean_.count(), input_data + offset,
						  data_mean_.cpu_data(), input_data + offset);
			}
		}

		if (has_mean_values) {
			CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
																					 "Specify either 1 mean_value or as many as channels: "
																					 << input_channels;
			if (mean_values_.size() == 1) {
				caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
			} else {
				for (int n = 0; n < input_num; ++n) {
					for (int c = 0; c < input_channels; ++c) {
						int offset = input_blob->offset(n, c);
						caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
										 input_data + offset);
					}
				}
			}
		}

		Dtype *transformed_data = transformed_blob->mutable_cpu_data();

		for (int n = 0; n < input_num; ++n) {
			int top_index_n = n * channels;
			int data_index_n = n * channels;
			for (int c = 0; c < channels; ++c) {
				int top_index_c = (top_index_n + c) * height;
				int data_index_c = (data_index_n + c) * input_height + h_off;
				for (int h = 0; h < height; ++h) {
					int top_index_h = (top_index_c + h) * width;
					int data_index_h = (data_index_c + h) * input_width + w_off;
					if (do_mirror) {
						int top_index_w = top_index_h + width - 1;
						for (int w = 0; w < width; ++w) {
							if (param_.is_flow() && c % 2 == 0)
								transformed_data[top_index_w - w] = 255 - input_data[data_index_h + w];
							else
								transformed_data[top_index_w - w] = input_data[data_index_h + w];
						}
					} else {
						for (int w = 0; w < width; ++w) {
							transformed_data[top_index_h + w] = input_data[data_index_h + w];
						}
					}
				}
			}
		}
		if (scale != Dtype(1)) {
			DLOG(INFO) << "Scale: " << scale;
			caffe_scal(size, scale, transformed_data);
		}
	}

	template<typename Dtype>
	vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum &datum) {
		if (datum.encoded()) {
			CHECK(!(param_.force_color() && param_.force_gray()))
			<< "cannot set both force_color and force_gray";
			cv::Mat cv_img;
			if (param_.force_color() || param_.force_gray()) {
				// If force_color then decode in color otherwise decode in gray.
				cv_img = DecodeDatumToCVMat(datum, param_.force_color());
			} else {
				cv_img = DecodeDatumToCVMatNative(datum);
			}
			// InferBlobShape using the cv::image.
			return InferBlobShape(cv_img);
		}

		const int crop_size = param_.crop_size();
		const int datum_channels = datum.channels();
		const int datum_height = datum.height();
		const int datum_width = datum.width();
		// Check dimensions.
		CHECK_GT(datum_channels, 0);
		CHECK_GE(datum_height, crop_size);
		CHECK_GE(datum_width, crop_size);
		// Build BlobShape.
		vector<int> shape(4);
		shape[0] = 1;
		shape[1] = datum_channels;
		shape[2] = (crop_size) ? crop_size : datum_height;
		shape[3] = (crop_size) ? crop_size : datum_width;
		return shape;
	}

	template<typename Dtype>
	vector<int> DataTransformer<Dtype>::InferBlobShape(
			const vector<Datum> &datum_vector) {
		const int num = datum_vector.size();
		CHECK_GT(num, 0) << "There is no datum to in the vector";
		// Use first datum in the vector to InferBlobShape.
		vector<int> shape = InferBlobShape(datum_vector[0]);
		// Adjust num to the size of the vector.
		shape[0] = num;
		return shape;
	}

	template<typename Dtype>
	vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat &cv_img) {
		const int crop_size = param_.crop_size();
		const int img_channels = cv_img.channels();
		const int img_height = cv_img.rows;
		const int img_width = cv_img.cols;
		// Check dimensions.
		CHECK_GT(img_channels, 0);
		if (!org_size_proc_) {
			CHECK_GE(img_height, crop_size);
			CHECK_GE(img_width, crop_size);
		}
		// Build BlobShape.
		vector<int> shape(4);
		shape[0] = 1;
		shape[1] = img_channels;
		shape[2] = (crop_size) ? crop_size : img_height;
		shape[3] = (crop_size) ? crop_size : img_width;
		return shape;
	}

	template<typename Dtype>
	vector<int> DataTransformer<Dtype>::InferBlobShape(
			const vector<cv::Mat> &mat_vector) {
		const int num = mat_vector.size();
		CHECK_GT(num, 0) << "There is no cv_img to in the vector";
		// Use first cv_img in the vector to InferBlobShape.
		vector<int> shape = InferBlobShape(mat_vector[0]);
		// Adjust num to the size of the vector.
		shape[0] = num;
		return shape;
	}

	template<typename Dtype>
	void DataTransformer<Dtype>::InitRand() {
		const bool needs_rand = param_.mirror() ||
								(phase_ == TRAIN &&
								 (param_.crop_size() || (param_.crop_width() && param_.crop_height())));
		if (needs_rand) {
			const unsigned int rng_seed = caffe_rng_rand();
			rng_.reset(new Caffe::RNG(rng_seed));
		} else {
			rng_.reset();
		}
	}

	template<typename Dtype>
	int DataTransformer<Dtype>::Rand(int n) {
		CHECK(rng_);
		CHECK_GT(n, 0);
		caffe::rng_t *rng =
				static_cast<caffe::rng_t *>(rng_->generator());
		return ((*rng)() % n);
	}

	INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
