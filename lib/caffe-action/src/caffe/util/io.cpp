#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include <string>
//#include <ifstream>
#include "boost/algorithm/string/split.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/lexical_cast.hpp"
#include <map>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#ifdef USE_MEMCACHED
#include "MemcachedClient.h"
#endif
const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.
using namespace cv;
using namespace std;

namespace caffe {

	using google::protobuf::io::FileInputStream;
	using google::protobuf::io::FileOutputStream;
	using google::protobuf::io::ZeroCopyInputStream;
	using google::protobuf::io::CodedInputStream;
	using google::protobuf::io::ZeroCopyOutputStream;
	using google::protobuf::io::CodedOutputStream;
	using google::protobuf::Message;
	vector<int> coor_vec;

	bool ReadProtoFromTextFile(const char *filename, Message *proto) {
		int fd = open(filename, O_RDONLY);
		CHECK_NE(fd, -1) << "File not found: " << filename;
		FileInputStream *input = new FileInputStream(fd);
		bool success = google::protobuf::TextFormat::Parse(input, proto);
		delete input;
		close(fd);
		return success;
	}

	void WriteProtoToTextFile(const Message &proto, const char *filename) {
		int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
		FileOutputStream *output = new FileOutputStream(fd);
		CHECK(google::protobuf::TextFormat::Print(proto, output));
		delete output;
		close(fd);
	}

	bool ReadProtoFromBinaryFile(const char *filename, Message *proto) {
		int fd = open(filename, O_RDONLY);
		CHECK_NE(fd, -1) << "File not found: " << filename;
		ZeroCopyInputStream *raw_input = new FileInputStream(fd);
		CodedInputStream *coded_input = new CodedInputStream(raw_input);
		coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

		bool success = proto->ParseFromCodedStream(coded_input);

		delete coded_input;
		delete raw_input;
		close(fd);
		return success;
	}

	void WriteProtoToBinaryFile(const Message &proto, const char *filename) {
		fstream output(filename, ios::out | ios::trunc | ios::binary);
		CHECK(proto.SerializeToOstream(&output));
	}

//	cv::Mat ReadImageToCVMat(const string &filename,
//							 const int height, const int width, const bool is_color) {
//		cv::Mat cv_img;
//		int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
//							CV_LOAD_IMAGE_GRAYSCALE);
//		cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
//		if (!cv_img_origin.data) {
//			LOG(ERROR) << "Could not open or find file " << filename;
//			return cv_img_origin;
//		}
//		if (height > 0 && width > 0) {
//			cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
//		} else {
//			cv_img = cv_img_origin;
//		}
//		return cv_img;
//	}



	cv::Mat ReadImageToCVMat(const string& filename,
							 const int height, const int width, const bool is_color,
							 int* img_height, int* img_width) {
#ifdef USE_MEMCACHED
		// get the configure file of the list of servers and of the client
  const string server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf";
  const string client_config_file = "/mnt/lustre/share/memcached_client/client.conf";
  auto mclient = MemcachedClient::GetInstance(server_list_config_file, client_config_file);
  vector<char> value;
  size_t value_length = 0;
  value_length = mclient->Get(filename, value);
  cv::Mat cv_img, cv_img_origin;
  if(value_length == 0){
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
  cv_img_origin = cv::imdecode(value, cv_read_flag);
#else
		// no cache
		cv::Mat cv_img, cv_img_origin;
		int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
							CV_LOAD_IMAGE_GRAYSCALE);
		cv_img_origin = cv::imread(filename, cv_read_flag);
		if (!cv_img_origin.data) {
			LOG(ERROR) << "Could not open or find file " << filename;
			return cv_img_origin;
		}
		// cache images in the memory
#if 0
		static std::unordered_map<string, string> filecache;
  static double mem_free_per = 1.0;
  static bool flag = false;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img, cv_img_origin;
  if (filecache.find(filename) == filecache.end()) {
    if (mem_free_per >= 0.1) {
      std::streampos size;
      fstream file(filename.c_str(), ios::in | ios::binary | ios::ate);
      if (file.is_open()) {
        size = file.tellg();
        std::string buffer(size, ' ');
        file.seekg(0, ios::beg);
        file.read(&buffer[0], size);
        file.close();
        filecache[filename] = buffer;
        std::vector<char> vec_buf(buffer.c_str(), buffer.c_str() + buffer.size());
        cv_img_origin = cv::imdecode(vec_buf, cv_read_flag);
      } else {
        LOG(ERROR) << "Could not open or find file " << filename;
        return cv_img_origin;
      }
      MEM_OCCUPY mem_stat;
      get_memoccupy((MEM_OCCUPY *)&mem_stat);
      mem_free_per = (double)mem_stat.free / (double)mem_stat.total;
    } else {
      if (!flag) {
        LOG(INFO) << "The number of images cached in the memory is: " << filecache.size();
        flag = true;
      }
      cv_img_origin = cv::imread(filename, cv_read_flag);
    }

    if (!cv_img_origin.data) {
      LOG(ERROR) << "Could not open or find file " << filename;
      return cv_img_origin;
    }
  } else {
    std::vector<char> vec_buf(filecache[filename].c_str(), filecache[filename].c_str() + filecache[filename].size());
    cv_img_origin = cv::imdecode(vec_buf, cv_read_flag);
  }
#endif
#endif

		if (height > 0 && width > 0) {
			int new_width = width;
			int new_height = height;
			if (height == 1 || width == 1) {
				float length = height > width ? height : width;
				if (cv_img_origin.rows < cv_img_origin.cols) { // height < width
					float scale = length / cv_img_origin.rows;
					new_width = scale * cv_img_origin.cols;
					new_height = length;
				}
				else { // width <= height
					float scale = length / cv_img_origin.cols;
					new_width = length;
					new_height = scale * cv_img_origin.rows;
				}
			}
			cv::resize(cv_img_origin, cv_img, cv::Size(new_width, new_height));
		} else {
			cv_img = cv_img_origin;
		}
		if (img_height != NULL) {
			*img_height = cv_img.rows;
		}
		if (img_width != NULL) {
			*img_width = cv_img.cols;
		}
		return cv_img;
	}

	cv::Mat ReadImageToCVMat(const string &filename,
							 const int height, const int width) {
		return ReadImageToCVMat(filename, height, width, true);
	}

	cv::Mat ReadImageToCVMat(const string &filename,
							 const bool is_color) {
		return ReadImageToCVMat(filename, 0, 0, is_color);
	}

	cv::Mat ReadImageToCVMat(const string &filename) {
		return ReadImageToCVMat(filename, 0, 0, true);
	}

// Do the file extension and encoding match?
	static bool matchExt(const std::string &fn,
						 std::string en) {
		size_t p = fn.rfind('.');
		std::string ext = p != fn.npos ? fn.substr(p) : fn;
		std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
		std::transform(en.begin(), en.end(), en.begin(), ::tolower);
		if (ext == en)
			return true;
		if (en == "jpg" && ext == "jpeg")
			return true;
		return false;
	}

	bool ReadImageToDatum(const string &filename, const int label,
						  const int height, const int width, const bool is_color,
						  const std::string &encoding, Datum *datum) {
		cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
		if (cv_img.data) {
			if (encoding.size()) {
				if ((cv_img.channels() == 3) == is_color && !height && !width &&
					matchExt(filename, encoding))
					return ReadFileToDatum(filename, label, datum);
				std::vector<uchar> buf;
				cv::imencode("." + encoding, cv_img, buf);
				datum->set_data(std::string(reinterpret_cast<char *>(&buf[0]),
											buf.size()));
				datum->set_label(label);
				datum->set_encoded(true);
				return true;
			}
			CVMatToDatum(cv_img, datum);
			datum->set_label(label);
			return true;
		} else {
			return false;
		}
	}

	bool ReadFileToDatum(const string &filename, const int label,
						 Datum *datum) {
		std::streampos size;

		fstream file(filename.c_str(), ios::in | ios::binary | ios::ate);
		if (file.is_open()) {
			size = file.tellg();
			std::string buffer(size, ' ');
			file.seekg(0, ios::beg);
			file.read(&buffer[0], size);
			file.close();
			datum->set_data(buffer);
			datum->set_label(label);
			datum->set_encoded(true);
			return true;
		} else {
			return false;
		}
	}

	cv::Mat DecodeDatumToCVMatNative(const Datum &datum) {
		cv::Mat cv_img;
		CHECK(datum.encoded()) << "Datum not encoded";
		const string &data = datum.data();
		std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
		cv_img = cv::imdecode(vec_data, -1);
		if (!cv_img.data) {
			LOG(ERROR) << "Could not decode datum ";
		}
		return cv_img;
	}

	cv::Mat DecodeDatumToCVMat(const Datum &datum, bool is_color) {
		cv::Mat cv_img;
		CHECK(datum.encoded()) << "Datum not encoded";
		const string &data = datum.data();
		std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
		int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
							CV_LOAD_IMAGE_GRAYSCALE);
		cv_img = cv::imdecode(vec_data, cv_read_flag);
		if (!cv_img.data) {
			LOG(ERROR) << "Could not decode datum ";
		}
		return cv_img;
	}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
	bool DecodeDatumNative(Datum *datum) {
		if (datum->encoded()) {
			cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
			CVMatToDatum(cv_img, datum);
			return true;
		} else {
			return false;
		}
	}

	bool DecodeDatum(Datum *datum, bool is_color) {
		if (datum->encoded()) {
			cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
			CVMatToDatum(cv_img, datum);
			return true;
		} else {
			return false;
		}
	}

	void CVMatToDatum(const cv::Mat &cv_img, Datum *datum) {
		CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
		datum->set_channels(cv_img.channels());
		datum->set_height(cv_img.rows);
		datum->set_width(cv_img.cols);
		datum->clear_data();
		datum->clear_float_data();
		datum->set_encoded(false);
		int datum_channels = datum->channels();
		int datum_height = datum->height();
		int datum_width = datum->width();
		int datum_size = datum_channels * datum_height * datum_width;
		std::string buffer(datum_size, ' ');
		for (int h = 0; h < datum_height; ++h) {
			const uchar *ptr = cv_img.ptr<uchar>(h);
			int img_index = 0;
			for (int w = 0; w < datum_width; ++w) {
				for (int c = 0; c < datum_channels; ++c) {
					int datum_index = (c * datum_height + h) * datum_width + w;
					buffer[datum_index] = static_cast<char>(ptr[img_index++]);
				}
			}
		}
		datum->set_data(buffer);
	}

// Verifies format of data stored in HDF5 file and reshapes blob accordingly.
	template<typename Dtype>
	void hdf5_load_nd_dataset_helper(
			hid_t file_id, const char *dataset_name_, int min_dim, int max_dim,
			Blob<Dtype> *blob) {
		// Verify that the dataset exists.
		CHECK(H5LTfind_dataset(file_id, dataset_name_))
		<< "Failed to find HDF5 dataset " << dataset_name_;
		// Verify that the number of dimensions is in the accepted range.
		herr_t status;
		int ndims;
		status = H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);
		CHECK_GE(status, 0) << "Failed to get dataset ndims for " << dataset_name_;
		CHECK_GE(ndims, min_dim);
		CHECK_LE(ndims, max_dim);

		// Verify that the data format is what we expect: float or double.
		std::vector<hsize_t> dims(ndims);
		H5T_class_t class_;
		status = H5LTget_dataset_info(
				file_id, dataset_name_, dims.data(), &class_, NULL);
		CHECK_GE(status, 0) << "Failed to get dataset info for " << dataset_name_;
		CHECK_EQ(class_, H5T_FLOAT) << "Expected float or double data";

		vector<int> blob_dims(dims.size());
		for (int i = 0; i < dims.size(); ++i) {
			blob_dims[i] = dims[i];
		}
		blob->Reshape(blob_dims);
	}

	template<>
	void hdf5_load_nd_dataset<float>(hid_t file_id, const char *dataset_name_,
									 int min_dim, int max_dim, Blob<float> *blob) {
		hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
		herr_t status = H5LTread_dataset_float(
				file_id, dataset_name_, blob->mutable_cpu_data());
		CHECK_GE(status, 0) << "Failed to read float dataset " << dataset_name_;
	}

	template<>
	void hdf5_load_nd_dataset<double>(hid_t file_id, const char *dataset_name_,
									  int min_dim, int max_dim, Blob<double> *blob) {
		hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
		herr_t status = H5LTread_dataset_double(
				file_id, dataset_name_, blob->mutable_cpu_data());
		CHECK_GE(status, 0) << "Failed to read double dataset " << dataset_name_;
	}

	template<>
	void hdf5_save_nd_dataset<float>(
			const hid_t file_id, const string &dataset_name, const Blob<float> &blob) {
		hsize_t dims[HDF5_NUM_DIMS];
		dims[0] = blob.num();
		dims[1] = blob.channels();
		dims[2] = blob.height();
		dims[3] = blob.width();
		herr_t status = H5LTmake_dataset_float(
				file_id, dataset_name.c_str(), HDF5_NUM_DIMS, dims, blob.cpu_data());
		CHECK_GE(status, 0) << "Failed to make float dataset " << dataset_name;
	}

	template<>
	void hdf5_save_nd_dataset<double>(
			const hid_t file_id, const string &dataset_name, const Blob<double> &blob) {
		hsize_t dims[HDF5_NUM_DIMS];
		dims[0] = blob.num();
		dims[1] = blob.channels();
		dims[2] = blob.height();
		dims[3] = blob.width();
		herr_t status = H5LTmake_dataset_double(
				file_id, dataset_name.c_str(), HDF5_NUM_DIMS, dims, blob.cpu_data());
		CHECK_GE(status, 0) << "Failed to make double dataset " << dataset_name;
	}

	bool
	ReadSegDataToDatum(const string &img_filename, const string &label_filename, Datum *datum_data, Datum *datum_label,
					   bool is_color) {

		string *datum_data_string, *datum_label_string;

		int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
							CV_LOAD_IMAGE_GRAYSCALE);


		cv::Mat cv_img = cv::imread(img_filename, cv_read_flag);
		cv::Mat cv_label = cv::imread(label_filename, CV_LOAD_IMAGE_GRAYSCALE);

		if (!cv_img.data || !cv_label.data) {
			LOG(ERROR) << "Could not load file " << label_filename;
			return false;
		}

		int num_channels = (is_color ? 3 : 1);

		datum_data->set_channels(num_channels);
		datum_data->set_height(cv_img.rows);
		datum_data->set_width(cv_img.cols);
		datum_data->clear_data();
		datum_data->clear_float_data();
		datum_data_string = datum_data->mutable_data();

		//TODO: change datum channel into 3 and insert the
		datum_label->set_channels(1);
		datum_label->set_height(cv_label.rows);
		datum_label->set_width(cv_label.cols);
		datum_label->clear_data();
		datum_label->clear_float_data();
		datum_label_string = datum_label->mutable_data();


		if (is_color) {
			for (int c = 0; c < num_channels; ++c) {
				for (int h = 0; h < cv_img.rows; ++h) {
					for (int w = 0; w < cv_img.cols; ++w) {
						datum_data_string->push_back(
								static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
					}
				}
			}
		} else {  // Faster than repeatedly testing is_color for each pixel w/i loop
			for (int h = 0; h < cv_img.rows; ++h) {
				for (int w = 0; w < cv_img.cols; ++w) {
					datum_data_string->push_back(
							static_cast<char>(cv_img.at<uchar>(h, w)));
				}
			}
		}

		for (int h = 0; h < cv_label.rows; ++h) {
			for (int w = 0; w < cv_label.cols; ++w) {
				datum_label_string->push_back(
						static_cast<char>(cv_label.at<uchar>(h, w)));
			}
		}
		return true;
	}


	bool ReadSegmentRGBToDatum(const string &filename, const int label,
							   const vector<int> offsets, const int height, const int width, const int length,
							   Datum *datum, bool is_color,
							   const char *name_pattern) {
		cv::Mat cv_img;
		string *datum_string;
		char tmp[30];
		int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
							CV_LOAD_IMAGE_GRAYSCALE);
		for (int i = 0; i < offsets.size(); ++i) {
			int offset = offsets[i];
			for (int file_id = 1; file_id < length + 1; ++file_id) {
				sprintf(tmp, name_pattern, int(file_id + offset));
				string filename_t = filename + "/" + tmp; //  frame img_path

#ifdef USE_MEMCACHED
				// get the configure file of the list of servers and of the client
		  const string server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf";
		  const string client_config_file = "/mnt/lustre/share/memcached_client/client.conf";
		  auto mclient = MemcachedClient::GetInstance(server_list_config_file, client_config_file);
		  vector<char> value;
		  size_t value_length = 0;
		  value_length = mclient->Get(filename_t, value);
		  cv::Mat cv_img, cv_img_origin;
		  if(value_length == 0){
			LOG(ERROR) << "Could not open or find file " << filename;
			return false;
		  }
		  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
		  cv_img_origin = cv::imdecode(value, cv_read_flag);
#else
				cv::Mat cv_img_origin = cv::imread(filename_t, cv_read_flag);
				if (!cv_img_origin.data) {
					LOG(ERROR) << "Could not load file " << filename_t;
					return false;
				}
#endif
				if (height > 0 && width > 0) {
					cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
				} else {
					cv_img = cv_img_origin;
				}
				int num_channels = (is_color ? 3 : 1);
				if (file_id == 1 && i == 0) {
					datum->set_channels(num_channels * length * offsets.size());
					datum->set_height(cv_img.rows);
					datum->set_width(cv_img.cols);
					datum->set_label(label);
					datum->clear_data();
					datum->clear_float_data();
					datum_string = datum->mutable_data();
				}
				if (is_color) {
					for (int c = 0; c < num_channels; ++c) {
						for (int h = 0; h < cv_img.rows; ++h) {
							for (int w = 0; w < cv_img.cols; ++w) {
								datum_string->push_back(
										static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
							}
						}
					}
				} else {  // Faster than repeatedly testing is_color for each pixel w/i loop
					for (int h = 0; h < cv_img.rows; ++h) {
						for (int w = 0; w < cv_img.cols; ++w) {
							datum_string->push_back(
									static_cast<char>(cv_img.at<uchar>(h, w)));
						}
					}
				}
			}
		}
		return true;
	}

	bool readSingleRGBFlowToDatum(bool &is_color, string *name_patterns, char *tmp, const string &filename,
								  int &i, int &offset, int &file_id, const int &length, int &pattern_id,
								  const int &height, const int &width, cv::Mat &cv_img, Datum *datum,
								  string *datum_string, const vector<int> &offsets, const int &label) {

		return true;
	}


	void visualize_datum(const string &transformed_data, const int width, const int height, string seg_id) {
		cv::Mat img1(cv::Size(width, height), CV_8UC3), img2(cv::Size(width, height), CV_8UC3),
				flow_x(cv::Size(width, height), CV_8UC1), flow_y(cv::Size(width, height), CV_8UC1);
		for (int c = 0; c < 24; c++) {
			if (c < 9) {
				if (c > 0 && c % 3 == 0) {
					std::stringstream ss_img1, ss_img2;
					ss_img1 << "visualize/" << seg_id.substr(0, seg_id.npos - 5) << width << "_" << height << "_" << c
							<< "_img1.jpg";
					ss_img2 << "visualize/" << seg_id.substr(0, seg_id.npos - 5) << width << "_" << height << "_" << c
							<< "_img2.jpg";

					cv::imwrite(ss_img1.str(), img1);
					cv::imwrite(ss_img2.str(), img2);
					LOG(INFO) << "IMG:" << ss_img1.str() << " WRITTEN";
				}
				for (int h = 0; h < height; h++) {
					for (int w = 0; w < width; w++) {
						int cur_datum_offset = c * height * width + h * width + w;
						int sec_datum_offset = (c + 9) * height * width + h * width + w;
						img1.at<cv::Vec3b>(h, w)[c % 3] = static_cast<uint8_t>(transformed_data[cur_datum_offset]);
						img2.at<cv::Vec3b>(h, w)[c % 3] = static_cast<uint8_t>(transformed_data[sec_datum_offset]);
					}
				}

			} else if (c >= 18 && c <= 20) {
				for (int h = 0; h < height; h++) {
					for (int w = 0; w < width; w++) {
						int cur_datum_offset = c * height * width + h * width + w;
						int sec_datum_offset = (c + 3) * height * width + h * width + w;
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


	bool ReadSegmentRGBFlowToDatum(const string &filename, const int label,
								   const vector<int> offsets, const int height, const int width, const int length,
								   Datum *datum, bool is_color,
								   const char *name_pattern) {
		cv::Mat cv_img;
		string *datum_string;
		char tmp[30];
		int num_channels = 3 * 2 + 2;
//		LOG(INFO) << "Loading RGB_FLOW...";
		// permute the data in the sequence like following:
		// | (img1) num_segments(3) * num_channel_per_img(3) * width * height |
		// | (img2) num_segments(3) * num_channel_per_img(3) * width * height |
		// | (flowx) num_segments(3) * num_channel_per_img(1) * width * height |
		// | (flowy) num_segments(3) * num_channel_per_img(1) * width * height |
		// permutation in whole: | img1(9) | img2(9) | flowx(3) | flowy(3) |, the num in brace is num_channel for each slice
		string name_patterns[] = {"img_prev_%05d.jpg", "img_%05d.jpg", "flow_x_%05d.jpg", "flow_y_%05d.jpg"};
		for (int pattern_id = 0; pattern_id < 4; pattern_id++) {
			is_color = name_patterns[pattern_id].find("img") !=
					   std::string::npos; // check the reading image is rgb | flow
			for (int i = 0; i < offsets.size(); ++i) {
				int offset = offsets[i]; // it may have multiple segments
				for (int file_id = 1; file_id < length + 1; ++file_id) {
					sprintf(tmp, name_patterns[pattern_id].c_str(), int(file_id + offset));
					string filename_t = filename + "/" + tmp; //  frame img_path: folder_name/image_%04d.jpg
					int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
					cv::Mat cv_img_origin = cv::imread(filename_t, cv_read_flag);
					if (!cv_img_origin.data) {
						LOG(ERROR) << "Could not load file " << filename_t;
						return false;
					}
//					cv::imwrite("visualize/"+string(tmp), cv_img_origin);
					if (height > 0 && width > 0) {
						cv::resize(cv_img_origin, cv_img, cv::Size(width, height)); // TODO: padding may be better
					} else {
						cv_img = cv_img_origin;
					}
//					cv::imwrite("visualize/" + string(tmp), cv_img);
					if (file_id == 1 && i == 0 && pattern_id == 0) { // only initialize the datum at the first time
						// each segment should contains 2 RGB images (6 channels), 2 flow images (2 channels)

						datum->set_channels(num_channels * length * offsets.size());
						datum->set_height(cv_img.rows);
						datum->set_width(cv_img.cols);
						datum->set_label(label);
						datum->clear_data();
						datum->clear_float_data();
						datum_string = datum->mutable_data();
					}
					if (is_color) {
						for (int c = 0; c < 3; ++c) {
							for (int h = 0; h < cv_img.rows; ++h) {
								for (int w = 0; w < cv_img.cols; ++w) {
									datum_string->push_back(static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
								}
							}
						}
					} else {  // Faster than repeatedly testing is_color for each pixel w/i loop
						for (int h = 0; h < cv_img.rows; ++h) {
							for (int w = 0; w < cv_img.cols; ++w) {
								datum_string->push_back(static_cast<char>(cv_img.at<uchar>(h, w)));
							}
						}
					}
				}
			}
		}

//		visualize_datum(datum->data(), cv_img.cols, cv_img.rows, string(tmp));
//		LOG(INFO) << "RGB_FLOW Loaded!";
		return true;
	}


	bool ReadSegmentFlowToDatum(const string &filename, const int label,
								const vector<int> offsets, const int height, const int width, const int length,
								Datum *datum,
								const char *name_pattern) {
		cv::Mat cv_img_x, cv_img_y;
		string *datum_string;
		char tmp[30];
		for (int i = 0; i < offsets.size(); ++i) {
			int offset = offsets[i];
			for (int file_id = 1; file_id < length + 1; ++file_id) {
				sprintf(tmp, name_pattern, 'x', int(file_id + offset));
				string filename_x = filename + "/" + tmp;
//				cv::Mat cv_img_origin_x = cv::imread(filename_x, CV_LOAD_IMAGE_GRAYSCALE);
				sprintf(tmp, name_pattern, 'y', int(file_id + offset));
				string filename_y = filename + "/" + tmp;
//				cv::Mat cv_img_origin_y = cv::imread(filename_y, CV_LOAD_IMAGE_GRAYSCALE);

#ifdef USE_MEMCACHED
				// get the configure file of the list of servers and of the client
		  const string server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf";
		  const string client_config_file = "/mnt/lustre/share/memcached_client/client.conf";
		  auto mclient = MemcachedClient::GetInstance(server_list_config_file, client_config_file);
		  vector<char> value_x, value_y;
		  size_t value_length_x = 0;
		  size_t value_length_y = 0;
		  value_length_x = mclient->Get(filename_x, value_x);
		  value_length_y = mclient->Get(filename_y, value_y);
		  cv::Mat cv_img_origin_x, cv_img_origin_y;
		  if(value_length_x == 0 || value_length_y == 0){
			LOG(ERROR) << "Could not open or find file " << filename_x;
			return false;
		  }
//		  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
		  cv_img_origin_x = cv::imdecode(value_x, CV_LOAD_IMAGE_GRAYSCALE);
		  cv_img_origin_y = cv::imdecode(value_y, CV_LOAD_IMAGE_GRAYSCALE);
#else
				cv::Mat cv_img_origin_x = cv::imread(filename_x, CV_LOAD_IMAGE_GRAYSCALE);
				cv::Mat cv_img_origin_y = cv::imread(filename_y, CV_LOAD_IMAGE_GRAYSCALE);
				if (!cv_img_origin_x.data || !cv_img_origin_y.data) {
					LOG(ERROR) << "Could not load file " << filename_x;
					return false;
				}
#endif
				if (!cv_img_origin_x.data || !cv_img_origin_y.data) {
					LOG(ERROR) << "Could not load file " << filename_x << " or " << filename_y;
					return false;
				}
				if (height > 0 && width > 0) {
					cv::resize(cv_img_origin_x, cv_img_x, cv::Size(width, height));
					cv::resize(cv_img_origin_y, cv_img_y, cv::Size(width, height));
				} else {
					cv_img_x = cv_img_origin_x;
					cv_img_y = cv_img_origin_y;
				}
				if (file_id == 1 && i == 0) {
					int num_channels = 2;
					datum->set_channels(num_channels * length * offsets.size());
					datum->set_height(cv_img_x.rows);
					datum->set_width(cv_img_x.cols);
					datum->set_label(label);
					datum->clear_data();
					datum->clear_float_data();
					datum_string = datum->mutable_data();
				}
				for (int h = 0; h < cv_img_x.rows; ++h) {
					for (int w = 0; w < cv_img_x.cols; ++w) {
						datum_string->push_back(static_cast<char>(cv_img_x.at<uchar>(h, w)));
					}
				}
				for (int h = 0; h < cv_img_y.rows; ++h) {
					for (int w = 0; w < cv_img_y.cols; ++w) {
						datum_string->push_back(static_cast<char>(cv_img_y.at<uchar>(h, w)));
					}
				}
//				visualize(datum, width, height, file_id);
			}
		}
		return true;
	}


	void
	LoadJoints(const std::string &filename, map<std::string, vector<vector<int> > > &person_map, std::string prefix) {
		//This part of code is extremely memory costing
		std::ifstream fjoints(filename.c_str());
		std::string line;
		std::vector<std::string> line_vec;
		while (getline(fjoints, line)) {
			std::string img_path;
			std::vector<int> joints;
			boost::split(line_vec, line, boost::is_any_of(" "));
			img_path = prefix + line_vec[0];
//			LOG(INFO) << "loading img: " << img_path;
			for (int jid = 3; jid < line_vec.size(); jid++) {
				joints.push_back(atoi(line_vec[jid].c_str()));
//				int x = atoi(line_vec[jid].c_str());
//				int y = atoi(line_vec[jid + 1].c_str());
//				int is_visible = atoi(line_vec[jid + 2].c_str());
			}
			if (!person_map.count(img_path)) {
				vector<vector<int> > person_vec;
				person_vec.push_back(joints);
				person_map[img_path] = person_vec;
			} else {
				person_map[img_path].push_back(joints);
			}


//			int num_people;
		}
	}

	void putPointGuassMap(Mat &entryX, Point2f &center, int g_x, int g_y, float sigma, float peak, float stride) {
		float x = g_x;
		float y = g_y;
		float d2 = (x - center.x) * (x - center.x) + (y - center.y) * (y - center.y);
		float exponent = d2 / 2.0 / sigma / sigma;
		if (exponent > (4.6052 / stride / stride)) { //ln(100) = -ln(1%)
			return;
		}
		entryX.at<uchar>(g_y, g_x) += static_cast<uchar>(peak); // exp(-exponent) *
		if (entryX.at<uchar>(g_y, g_x) > peak)
			entryX.at<uchar>(g_y, g_x) = static_cast<uchar>(peak);
	}

	void putPointVecMap(Mat &entryX, Mat &count, Point2f &centerA, Point2f &bc, int g_x, int g_y,
						float thre, float peak_ratio) {
		Point2f ba;
		ba.x = g_x - centerA.x;
		ba.y = g_y - centerA.y;
		float dist = std::abs(ba.x * bc.y - ba.y * bc.x);
//				LOG(INFO) << "DIST: " << dist;
		if (dist <= thre) {
			int cnt = count.at<uchar>(g_y, g_x);
			if (cnt == 0) {
				entryX.at<uchar>(g_y, g_x) = static_cast<uchar>(peak_ratio) >= 0 ? static_cast<uchar>(peak_ratio) :
											 (unsigned char) 0; // bc.x *
			} else {
				entryX.at<uchar>(g_y, g_x) = static_cast<uchar>((static_cast<float >(entryX.at<uchar>(g_y, g_x)) * cnt
																 + peak_ratio) / (cnt + 1)) >= 0 ?
											 static_cast<uchar>((static_cast<float >(entryX.at<uchar>(g_y, g_x)) * cnt
																 + peak_ratio) / (cnt + 1))
																								 : (unsigned char) 0; // bc.x *
				count.at<uchar>(g_y, g_x) = (unsigned char) cnt + 1;
			}
		}
	}


	void putGaussianMaps(Mat &entryX, Point2f center, int stride, int grid_x, int grid_y, float sigma, float peak) {
		//LOG(INFO) << "putGaussianMaps here we start for " << center.x << " " << center.y;
		float start = stride / 2.0 - 0.5; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
		for (int g_y = 0; g_y < grid_y; g_y++) {
			for (int g_x = 0; g_x < grid_x; g_x++) {
				float x = start + g_x * stride;
				float y = start + g_y * stride;
				float d2 = (x - center.x) * (x - center.x) + (y - center.y) * (y - center.y);
				float exponent = d2 / 2.0 / sigma / sigma;
				if (exponent > 4.6052) { //ln(100) = -ln(1%)
					continue;
				}
				entryX.at<uchar>(g_y, g_x) += static_cast<uchar>(peak); // exp(-exponent) *
				if (entryX.at<uchar>(g_y, g_x) > peak)
					entryX.at<uchar>(g_y, g_x) = static_cast<uchar>(peak);
			}
		}
	}

	void putAllMaps(Mat &entryX, Mat &count, Point2f &centerA, Point2f &centerB, bool isValidA, bool isValidB,
					float sigmaA, float sigmaB, float stride, int grid_x, int grid_y, float thre, float peak_ratio) {

		centerB = centerB * (1 / stride);
		centerA = centerA * (1 / stride);
		Point2f bc = centerB - centerA;
		int min_x = std::max(int(round(std::min(centerA.x, centerB.x) - thre)), 0);
		int max_x = std::min(int(round(std::max(centerA.x, centerB.x) + thre)), grid_x);

		int min_y = std::max(int(round(std::min(centerA.y, centerB.y) - thre)), 0);
		int max_y = std::min(int(round(std::max(centerA.y, centerB.y) + thre)), grid_y);

		float norm_bc = sqrt(bc.x * bc.x + bc.y * bc.y);
		bc.x = bc.x / norm_bc;
		bc.y = bc.y / norm_bc;


		for (int g_y = 0; g_y < grid_y; g_y++) {
			for (int g_x = 0; g_x < grid_x; g_x++) {
				bool is_on_limb = g_x >= min_x && g_x <= max_x && g_y >= min_y && g_y <= max_y;
				if (isValidA)
					putPointGuassMap(entryX, centerA, g_x, g_y, sigmaA, peak_ratio, stride);
				if (isValidB)
					putPointGuassMap(entryX, centerB, g_x, g_y, sigmaB, peak_ratio, stride);
				if (isValidA && isValidB && is_on_limb)
					putPointVecMap(entryX, count, centerA, bc, g_x, g_y, thre, peak_ratio);
			}

		}
	}


	void putVecMaps(Mat &entryX, Mat &count, Point2f centerA, Point2f centerB, float stride, int grid_x,
					int grid_y, int thre, float peak_ratio) {
		//int thre = 4;
		centerB = centerB * (1 / stride);
		centerA = centerA * (1 / stride);
		Point2f bc = centerB - centerA;
		int min_x = std::max(int(round(std::min(centerA.x, centerB.x) - thre)), 0);
		int max_x = std::min(int(round(std::max(centerA.x, centerB.x) + thre)), grid_x);

		int min_y = std::max(int(round(std::min(centerA.y, centerB.y) - thre)), 0);
		int max_y = std::min(int(round(std::max(centerA.y, centerB.y) + thre)), grid_y);

		float norm_bc = sqrt(bc.x * bc.x + bc.y * bc.y);
		bc.x = bc.x / norm_bc;
		bc.y = bc.y / norm_bc;

		for (int g_y = min_y; g_y < max_y; g_y++) {
			for (int g_x = min_x; g_x < max_x; g_x++) {
				Point2f ba;
				ba.x = g_x - centerA.x;
				ba.y = g_y - centerA.y;
				float dist = std::abs(ba.x * bc.y - ba.y * bc.x);
//				LOG(INFO) << "DIST: " << dist;
				if (dist <= thre) {
					int cnt = count.at<uchar>(g_y, g_x);
					if (cnt == 0) {
						entryX.at<uchar>(g_y, g_x) =
								static_cast<uchar>(peak_ratio) >= 0 ? static_cast<uchar>(peak_ratio) :
								(unsigned char) 0; // bc.x *
					} else {
						entryX.at<uchar>(g_y, g_x) =
								static_cast<uchar>((static_cast<float >(entryX.at<uchar>(g_y, g_x)) * cnt
													+ peak_ratio) / (cnt + 1)) >= 0 ?
								static_cast<uchar>((static_cast<float >(entryX.at<uchar>(g_y, g_x)) * cnt
													+ peak_ratio) / (cnt + 1)) : (unsigned char) 0; // bc.x *
						count.at<uchar>(g_y, g_x) = (unsigned char) cnt + 1;
					}
				}
			}
		}
	}

	void addROI(vector<int> &rois, int &id, int &top_w, int &top_h, int &w, int &h) {
//		LOG(INFO) << "ADDING: cnt:" << id<<", top_w:" << top_w<<", top_h:" <<top_h <<", w:" << w << ", h:" <<h;
		rois.push_back(id);
		rois.push_back(top_w);
		rois.push_back(top_h);
		rois.push_back(w);
		rois.push_back(h);
	}

	bool contain_joint(vector<int> joints, int joint_id) {
		return std::find(joints.begin(), joints.end(), joint_id) != joints.end();
	}

	bool calcLineBasedROI(vector<int> joints, vector<int> rois, int jid0, int jid1, int image_id,
						  float roi_h, float roi_w) {
//		const int jid0 = select_joints[0];
//		const int jid1 = select_joints[1];
		int is_valid0 = joints[jid0 * 3 + 2];
		int is_valid1 = joints[jid1 * 3 + 2];
		if (is_valid0 && is_valid1) {
			float edge = std::sqrt(float(joints[jid0 * 3] - joints[jid1 * 3])
								   * float(joints[jid0 * 3] - joints[jid1 * 3])
								   + float(joints[jid0 * 3 + 1] - joints[jid1 * 3 + 1])
									 * float(joints[jid0 * 3 + 1] - joints[jid1 * 3 + 1]));
			if (edge >= min(roi_h, roi_w)) {
				Point2f center = Point2f((joints[jid0 * 3] + joints[jid1 * 3]) / 2,
										 (joints[jid0 * 3 + 1] + joints[jid1 * 3 + 1]) / 2);
				int l_w = int(center.x - edge / 2);
				int l_h = int(center.y - edge / 2);
				int egde_to_save = int(edge);
				addROI(rois, image_id, l_w, l_h, egde_to_save, egde_to_save);
				return true;
			}
		}
		return false;
	}

	void
	generateLimb(cv::Mat &img_dst, vector<vector<int> > person_joints, float scale_x, float scale_y, float stride) {
		/**
		 * Joint Sequence:
		 *   0    1    2    3    4    5    6    7    8    9    10    11   12   13   14   15   16   17
		 * neck rsho rarm rwri lsho larm lwri rhip rknee rank lhip lknee lank reye leye rear lear nose
		 */

		//limb sequence
//		int np = 18;
//		float stride = 8.0;
		int rezX = img_dst.cols;
		int rezY = img_dst.rows;
		int mid_1[19] = {2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16};
		int mid_2[19] = {9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18};
		float sigma_seq[19] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
							   1, 1};//3,3,9,16,3,9,16,3,3,3,3,3,3,3,3,3,3,3

		float thre = float(0 / stride); // 16.0
		// link limbs
		for (int i = 0; i < 19; i++) {
			cv::Mat count = cv::Mat::zeros(rezY, rezX, CV_8UC1);
//			LOG(INFO) << "Containing " << person_joints.size() << " people";
			for (int k = 0; k < person_joints.size(); k++) {
				vector<int> &joints = person_joints[k];
				int is_visible0 = joints[3 * (mid_1[i] - 1) + 2];
				int is_visible1 = joints[3 * (mid_2[i] - 1) + 2];
				Point2f point0 = Point2f(joints[3 * (mid_1[i] - 1)] * scale_x,
										 joints[3 * (mid_1[i] - 1) + 1] * scale_y);
				Point2f point1 = Point2f(joints[3 * (mid_2[i] - 1)] * scale_x,
										 joints[3 * (mid_2[i] - 1) + 1] * scale_y);
				float sigma =
						sqrt((point0 - point1).x * (point0 - point1).x + (point0 - point1).y * (point0 - point1).y) / 4;
				if (mid_1[i] - 1 == 3 || mid_1[i] - 1 == 4 || mid_1[i] - 1 == 6 || mid_1[i] - 1 == 7 ||
					mid_2[i] - 1 == 3 || mid_2[i] - 1 == 4 || mid_2[i] - 1 == 6 || mid_2[i] - 1 == 7) {
					sigma = std::max(sigma, sigma_seq[mid_1[i] - 1]);
				}
//				LOG(INFO) << "Sigma: " << sigma;
				float sigma_0 = (is_visible0 && is_visible1) && (sigma > 0 && sigma <= sigma_seq[mid_1[i] - 1])
								? sigma : sigma_seq[mid_1[i] - 1];
				float sigma_1 = (is_visible0 && is_visible1) && (sigma > 0 && sigma <= sigma_seq[mid_1[i] - 1])
								? sigma : sigma_seq[mid_2[i] - 1];

				putAllMaps(img_dst, count, point0, point1, (is_visible0 > 0), (is_visible1 > 0),
						   sigma_0, sigma_1, stride, rezX, rezY, thre, 255.0);

//				if (is_visible0>0)
//					cv::putText(img_dst, boost::lexical_cast<string>(mid_1[i] - 1),point0, 1, 1, Scalar(128));
//				if(is_visible1>0)
//					cv::putText(img_dst, boost::lexical_cast<string>(mid_2[i] - 1),point1, 1, 1, Scalar(128));

			}
		}
	}


	vector<int> generateROI(vector<vector<int> > person_joints, vector<int> select_joints, float scale_x,
							float scale_y, float roi_w, float roi_h, int image_id) {
//		LOG(INFO) << "Got roi_w: " << roi_w << ", roi_h: " << roi_h << " with "
//				  << person_joints.size() << " people and " << select_joints.size() << " joints to select.";
		const int num_select_joints = select_joints.size();
		vector<int> rois;
		int top_w = 0, top_h = 0, w = 0, h = 0;

		for (int k = 0; k < person_joints.size(); k++) {
			vector<int> &joints = person_joints[k];
			if (num_select_joints == 2) {
				if (calcLineBasedROI(joints, rois, select_joints[0], select_joints[1], image_id,
									 roi_h, roi_w))
					continue;
			}

			// arms & wrists
//			if ((contain_joint(select_joints, 2) && contain_joint(select_joints, 3))) {
//				if (calcLineBasedROI(joints, rois, 2, 3, image_id, roi_h, roi_w)) {
//					select_joints.erase(std::remove(select_joints.begin(), select_joints.end(), 2),
//										select_joints.end());
//					select_joints.erase(std::remove(select_joints.begin(), select_joints.end(), 3),
//										select_joints.end());
//				}
//			}
//			if (contain_joint(select_joints, 5) && contain_joint(select_joints, 6)) {
//				if (calcLineBasedROI(joints, rois, 5, 6, image_id, roi_h, roi_w)) {
//					select_joints.erase(std::remove(select_joints.begin(), select_joints.end(), 5),
//										select_joints.end());
//					select_joints.erase(std::remove(select_joints.begin(), select_joints.end(), 6),
//										select_joints.end());
//				}
//			}

			// general case, generating bounding box from given key points
			for (int i = 0; i < num_select_joints; i++) {
				const int jid = select_joints[i];
				int &is_valid = joints[3 * jid + 2];
				if (is_valid > 0) {
					int x = int(joints[3 * jid] * scale_x);
					int y = int(joints[3 * jid + 1] * scale_y);
					if (x < 0 || y < 0)
						continue;
//					LOG(INFO) << "X, Y:" << x << ", " <<y;
//					LOG(INFO) << "top_w, top_h:" << top_w << ", " <<top_h;
					top_w = top_w == 0 && x > 0 ? x : x < top_w ? x : top_w;
					top_h = top_h == 0 && y > 0 ? y : y < top_h ? y : top_h;
					if (top_h || top_w) {
						w = w == 0 && x - top_w > 0 ? x - top_w : x - top_w > w ? x - top_w : w;
						h = h == 0 && y - top_h > 0 ? y - top_h : y - top_h > h ? y - top_h : h;
					}
				}
			}
			if ((top_h || top_w || w || h) && (w > roi_w || h > roi_h)) {

				int e = int(std::max(w, h));
				if (contain_joint(select_joints, 17) && joints[3 * 17 + 2] > 0) {
					top_w = int(joints[3 * 17] * scale_x - e / 2) >= 0 ? int(joints[3 * 17] * scale_x - e / 2) : 0;
					top_h = int(joints[3 * 17 + 1] * scale_y - e / 2) >= 0 ? int(joints[3 * 17 + 1] * scale_y - e / 2) : 0;
				}
				addROI(rois, image_id, top_w, top_h, e, e);
			}
		}
		return rois;
	}

	vector<int> merge_rois(vector<int> rois) {
		int min_x = 999, min_y = 999;
		int max_x = 0, max_y = 0;
		vector<int> roi;
		for (int roi_id = 0; roi_id < rois.size(); roi_id++){
			if (roi_id % 5 == 0) {
				continue;
			}
			if (roi_id % 5 == 1) min_x = rois[roi_id] < min_x ? rois[roi_id] : min_x;
			if (roi_id % 5 == 2) min_y = rois[roi_id] < min_y ? rois[roi_id] : min_y;
			if (roi_id % 5 == 3) max_x = rois[roi_id] + rois[roi_id - 2] > max_x ? rois[roi_id] + rois[roi_id - 2] : max_x;
			if (roi_id % 5 == 4) max_y = rois[roi_id] + rois[roi_id - 2] > max_y ? rois[roi_id] + rois[roi_id - 2] : max_y;
		}
		roi.push_back(0);
		roi.push_back(min_x);
		roi.push_back(min_y);
		roi.push_back(max_x);
		roi.push_back(max_y);
		return roi;
	}

	bool ReadSegmentRGBPoseToDatum(const string &filename, const int label, const vector<int> offsets, const int height,
								   const int width, const int length, Datum *datum, bool is_color, bool roi_pool_flag,
								   const char *name_pattern, map<std::string, vector<vector<int> > > &person_map, int item_id,
								   vector<int> select_joints, float stride, float roi_w, float roi_h) {

		cv::Mat cv_img;
		Datum temp_datum;
		string *datum_string;
		string *temp_datum_string;

		char tmp[30];
		int num_channels = 3;
//		LOG(INFO) << "Loading RGB_FLOW...";
		// permute the data in the sequence like following:
		// | (img1) num_segments(3) * num_channel_per_img(3) * width * height |
		// | (img2) num_segments(3) * num_channel_per_img(3) * width * height |
		// | (flowx) num_segments(3) * num_channel_per_img(1) * width * height |
		// | (flowy) num_segments(3) * num_channel_per_img(1) * width * height |
		// permutation in whole: | img1(9) | img2(9) | flowx(3) | flowy(3) |, the num in brace is num_channel for each slice
		string name_patterns[] = {"img_%05d.jpg"};

//		for (int pattern_id = 0; pattern_id < 1; pattern_id++) {
		for (int i = 0; i < offsets.size(); ++i) { // i is segment id
			int offset = offsets[i]; // it may have multiple segments
			for (int file_id = 1; file_id < length + 1; ++file_id) { // file_id is the file_id in a specific segment
				sprintf(tmp, name_patterns[0].c_str(), int(file_id + offset));
				string filename_t = filename + "/" + tmp; //  frame img_path: folder_name/image_%04d.jpg
				char *filename_abs_cstr;
				std::string filename_abs;
				filename_abs_cstr = realpath(filename_t.c_str(), NULL);
				if (filename_abs_cstr != NULL) {
					filename_abs = std::string(filename_abs_cstr);
//					LOG(INFO) << "filename_abs:" <<filename_abs;
					free(filename_abs_cstr);
				} else {
					LOG(ERROR) << "Cannot get valid abs file path";
				}

				int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
				vector<vector<int> > person_joints = person_map[filename_abs];
//				if (!person_joints.size())
//					LOG(INFO) << filename_abs << " has no joints.";

				cv::Mat cv_img_origin = cv::imread(filename_t, cv_read_flag);
				if (!cv_img_origin.data) {
					LOG(ERROR) << "Could not load file " << filename_t;
					return false;
				}
				float scale_x = 1;
				float scale_y = 1;
				if (height > 0 && width > 0) {
					cv::resize(cv_img_origin, cv_img, cv::Size(width, height)); // TODO: padding may be better
					scale_x = float(width / cv_img_origin.cols / 336.0);
					scale_y = float(height / cv_img_origin.rows / 256.0);
				} else {
					cv_img = cv_img_origin;
					scale_x = float(cv_img_origin.cols / 336.0);
					scale_y = float(cv_img_origin.rows / 256.0);
				}
				if (roi_pool_flag) {
					int image_id = int(item_id * offsets.size() + i);
					vector<int> rois = merge_rois(generateROI(person_joints, select_joints, scale_x, scale_y, roi_w, roi_h, image_id));

//					LOG(INFO) << "ROIS SIZE:" << rois.size();
					CHECK_EQ(rois.size() % 5, 0);

					/* TODO: Uncomment the following to visualize
					if(rois.size()){
						cv::Mat cv_pose(cv_img, cv::Rect(rois[1], rois[2],
														 rois[3] + rois[1] > cv_img.cols ? cv_img.cols - rois[1] : rois[3],
														 rois[4] + rois[2] > cv_img.rows ? cv_img.rows - rois[2] : rois[4]));
						cv::imwrite("visualize/pose_" + string(tmp), cv_pose);
					}
//					*/

					if (file_id == 1 && i == 0) { // only initialize the datum at the first time
						datum->set_channels(num_channels * length * offsets.size());
						datum->set_height(cv_img.rows);
						datum->set_width(cv_img.cols);
//						if (label == 77 || label == 19 || label == 0 || label == 1) {
//							datum->set_label(label);
//						} else {
//							datum->set_label(101); // others
//						}

						// Face Label Format
						if (label == 77) {
							datum->set_label(2);
						} else if (label == 19) {
							datum->set_label(3);
						} else if(label == 0 || label == 1) {
							datum->set_label(label);
						} else {
							datum->set_label(4); // others
						}
						//Arm Label Format

//						if (label == 98) {
//							datum->set_label(0);
//						} else if (label == 22) {
//							datum->set_label(1);
//						} else if (label == 23) {
//							datum->set_label(2);
//						} else if (label == 35) {
//							datum->set_label(3);
//						} else if (label == 44) {
//							datum->set_label(4);
//						} else if (label == 46) {
//							datum->set_label(5);
//						} else if (label == 55) {
//							datum->set_label(6);
//						} else if (label == 57) {
//							datum->set_label(7);
//						} else {
//							datum->set_label(8);
//						}

						datum->clear_data();
						datum->clear_float_data();
						datum_string = datum->mutable_data();
						coor_vec.clear();
//						temp_datum.set_channels(1 * length * offsets.size());
//						temp_datum.set_height(1);
//						temp_datum.set_width(5); // 1 roi at most, each with 5 roi coordinate
//						temp_datum->set_label(label);
//						temp_datum.clear_data();
//						temp_datum.clear_float_data();
//						temp_datum_string = temp_datum.mutable_data();
					}
					for (int c = 0; c < 3; ++c) {
						for (int h = 0; h < cv_img.rows; ++h) {
							for (int w = 0; w < cv_img.cols; ++w) {
								datum_string->push_back(static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
							}
						}
					}
					if (rois.size()) {
						for (int rid = 0; rid < rois.size(); rid++) {
							if (rid > 4) break;
//							LOG(INFO) << "Appending :"<<static_cast<uint8_t >(rois[rid]) << " from:" << rois[rid]
//									  << " with int cast: " << (int)static_cast<uint8_t >(rois[rid]);
							coor_vec.push_back(rois[rid]);
						}
					} else {
						for (int v = 0; v < 5; v++) {
							coor_vec.push_back(0);
						}
					}


				} else {
					cv::Mat cv_pose_init = cv::Mat::zeros(int(floor(cv_img.rows / stride)),
														  int(floor(cv_img.cols / stride)), CV_8UC1);
//					cv::imwrite("visualize/pose_init" + string(tmp), cv_pose);
					generateLimb(cv_pose_init, person_joints, scale_x, scale_y, stride);
//					cv::imwrite("visualize/pose_" + string(tmp), cv_pose_init);
					cv::Mat cv_pose;
					cv::resize(cv_pose_init, cv_pose, Size(cv_img.cols, cv_img.rows));

//					cv::imwrite("visualize/" + string(tmp), cv_img);
					if (file_id == 1 && i == 0) { // only initialize the datum at the first time
						datum->set_channels(num_channels * length * offsets.size());
						datum->set_height(cv_img.rows);
						datum->set_width(cv_img.cols);
						datum->set_label(label);
						datum->clear_data();
						datum->clear_float_data();
						datum_string = datum->mutable_data();


						temp_datum.set_channels(1 * length * offsets.size());
						temp_datum.set_height(cv_img.rows);
						temp_datum.set_width(cv_img.cols);
//						temp_datum->set_label(label);
						temp_datum.clear_data();
						temp_datum.clear_float_data();
						temp_datum_string = temp_datum.mutable_data();
					}
					for (int c = 0; c < 3; ++c) {
						for (int h = 0; h < cv_img.rows; ++h) {
							for (int w = 0; w < cv_img.cols; ++w) {
								datum_string->push_back(static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
							}
						}
					}
					for (int h = 0; h < cv_img.rows; ++h) {
						for (int w = 0; w < cv_img.cols; ++w) {
							temp_datum_string->push_back(static_cast<char>(cv_pose.at<uchar>(h, w)));
						}
					}
				}
			}
		}
//		}
		if (!roi_pool_flag) {
			datum_string->reserve(
					datum_string->size() + distance(temp_datum_string->begin(), temp_datum_string->end()));
			datum_string->insert(datum_string->end(), temp_datum_string->begin(), temp_datum_string->end());
		}

//		visualize_datum(datum->data(), cv_img.cols, cv_img.rows, string(tmp));
//		LOG(INFO) << "RGB_FLOW Loaded!";
		return true;
	}

}  // namespace caffe
