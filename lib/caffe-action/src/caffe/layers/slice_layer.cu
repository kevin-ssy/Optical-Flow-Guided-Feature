#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/contrib/contrib.hpp>

namespace caffe {

template <typename Dtype>
__global__ void Slice(const int nthreads, const Dtype* in_data,
    const bool forward, const int num_slices, const int slice_size,
    const int bottom_slice_axis, const int top_slice_axis,
    const int offset_slice_axis, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_slice_size = slice_size * top_slice_axis;
    const int slice_num = index / total_slice_size;
    const int slice_index = index % total_slice_size;
    const int bottom_index = slice_index +
        (slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;
    if (forward) {
      out_data[index] = in_data[bottom_index];
    } else {
      out_data[bottom_index] = in_data[index];
    }
  }
}
    template<typename Dtype>
    void visualize_top(Dtype *transformed_data, const int width, const int height,
                        const int num_segs, string seg_id){
      cv::Mat img1(cv::Size(width, height), CV_8UC3), img_pose(cv::Size(width, height), CV_8UC1);

      for (int n_seg = 0; n_seg < num_segs; n_seg++){
        for (int c =  3 * n_seg; c < 3 * (n_seg + 1); c++) {
          LOG(INFO) << "SEGS:" << num_segs;
          for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
              int cur_datum_offset = c * height * width + h * width + w;

              img1.at<cv::Vec3b>(h, w)[c % 3] = c == 0 ? static_cast<uint8_t>(transformed_data[cur_datum_offset]+ 104) :
                                                c == 1 ? static_cast<uint8_t>(transformed_data[cur_datum_offset] + 117) :
                                                static_cast<uint8_t>(transformed_data[cur_datum_offset] + 123);
//						LOG(INFO) << "CUR_PIXEL: "<<static_cast<uint8_t>(transformed_data[cur_datum_offset]);
            }
          }
          if (c % 3 == 2) {
            std::stringstream ss_img1;
            ss_img1 << "visualize/" << seg_id.substr(0, seg_id.npos - 2) << width << "_" << height << "_" << n_seg << "_img1.jpg";
            cv::imwrite(ss_img1.str(), img1);
            LOG(INFO) << "IMG:" << ss_img1.str() << " WRITTEN";
          }
        }
      }
    }
    template<typename Dtype>
    void visualize_pose(Dtype *transformed_data, const int width, const int height,
                        const int num_segs, string seg_id) {
      LOG(INFO) << "num_segs:" <<num_segs;
      LOG(INFO) << "WIDTH:" <<width;
      LOG(INFO) << "HEIGHT:" <<height;
      cv::Mat img1(cv::Size(width, height), CV_8UC3), img_pose(cv::Size(width, height), CV_8UC1);

      for (int n_seg = 0; n_seg < num_segs; n_seg++){
        for (int c =  3 * n_seg; c < 3 * (n_seg + 1); c++) {
          LOG(INFO) << "SEGS:" << num_segs;
          for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
              int cur_datum_offset = c * height * width + h * width + w;

              img1.at<cv::Vec3b>(h, w)[c % 3] = c == 0 ? static_cast<uint8_t>(transformed_data[cur_datum_offset]+ 104) :
                                                c == 1 ? static_cast<uint8_t>(transformed_data[cur_datum_offset] + 117) :
                                                static_cast<uint8_t>(transformed_data[cur_datum_offset] + 123);
//						LOG(INFO) << "CUR_PIXEL: "<<static_cast<uint8_t>(transformed_data[cur_datum_offset]);
            }
          }
          if (c % 3 == 2) {
            std::stringstream ss_img1;
            ss_img1 << "visualize/" << seg_id.substr(0, seg_id.npos - 2) << width << "_" << height << "_" << n_seg << "_img1.jpg";
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
        ss_flowx << "visualize/"<< seg_id.substr(0, seg_id.npos - 2) << width << "_" << height << "_" << n_seg << "_datum_pose.jpg";
        cv::imwrite(ss_flowx.str(), img_pose);

        cv::Mat img_wrighted(cv::Size(width, height), CV_8UC1);
        cv::Mat img_grey;
        cv::cvtColor(img1, img_grey, CV_BGR2GRAY);
        cv::addWeighted(img_grey, 0.5, img_pose, 0.5, 0, img_wrighted);

        std::stringstream ss_pose;
        ss_pose << "visualize/"<< seg_id.substr(0, seg_id.npos - 2) << width << "_" << height << "_" << n_seg << "_datum_pose_weighted.jpg";
        cv::imwrite(ss_pose.str(), img_wrighted);
        LOG(INFO) << "IMG:" << ss_pose.str() << " WRITTEN";
      }


    }

template <typename Dtype>
void SliceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//    int rand_id = rand() % 100000;
//    std::stringstream ss;
//    ss << rand_id<<"_slice_";
//    for (int n = 0; n < bottom[0]->shape(0); n++){
//    visualize_pose(bottom[0]->cpu_data(), bottom[0]->shape(3), bottom[0]->shape(2), 3, ss.str());
//}




  int offset_slice_axis = 0;
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  const bool kForward = true;
  for (int i = 0; i < top.size(); ++i) {
    Dtype* top_data = top[i]->mutable_gpu_data();
    const int top_slice_axis = top[i]->shape(slice_axis_);
    const int top_slice_size = top_slice_axis * slice_size_;
    const int nthreads = top_slice_size * num_slices_;
    Slice<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom_data, kForward, num_slices_, slice_size_,
        bottom_slice_axis, top_slice_axis, offset_slice_axis, top_data);
    offset_slice_axis += top_slice_axis;
  }

//for (int n = 0; n < top[0]->shape(0); n++){
//visualize_top(top[0]->cpu_data(), top[0]->shape(3), top[0]->shape(2), 3, ss.str()+"slice_top_");
//}
}

template <typename Dtype>
void SliceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  int offset_slice_axis = 0;
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  const bool kForward = false;
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    const int top_slice_axis = top[i]->shape(slice_axis_);
    const int top_slice_size = top_slice_axis * slice_size_;
    const int nthreads = top_slice_size * num_slices_;
    Slice<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, top_diff, kForward, num_slices_, slice_size_,
        bottom_slice_axis, top_slice_axis, offset_slice_axis, bottom_diff);
    offset_slice_axis += top_slice_axis;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SliceLayer);

}  // namespace caffe
