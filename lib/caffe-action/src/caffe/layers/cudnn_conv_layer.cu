#ifdef USE_CUDNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace caffe {

    template<typename Dtype>
    void visualize(const Dtype *data, int num, int channels, int width, int height, string img_key) {
      cv::Mat img1(cv::Size(width, height), CV_8UC3), img2(cv::Size(width, height), CV_8UC3);
//      LOG(INFO) << "Visualizing: " << channels << " " << width << " " << height;
//      cv::Mat flow_x(cv::Size(width, height), CV_8UC1), flow_y(cv::Size(width, height), CV_8UC1);
      if (channels == 6 && width == 256 && height == 256) {
        for (int n = 0; n < num; n++) {
          //img1
          for (int c = 0; c < 3; c++)
            for (int y = 0; y < height; y++)
              for (int x = 0; x < width; x++) {
                cv::Vec3b &rgb = img1.at<cv::Vec3b>(y, x);
                rgb[c] = (unsigned char) data[n * 6 * height * width + c * height * width + y * width + x];
              }
          //img2
          int img2_offset = n * 6 * height * width + 3 * height * width;
          for (int c = 0; c < 3; c++)
            for (int y = 0; y < height; y++)
              for (int x = 0; x < width; x++) {
                cv::Vec3b &rgb = img2.at<cv::Vec3b>(y, x);
                rgb[c] = (unsigned char) data[img2_offset + c * height * width + y * width + x];
              }

          std::stringstream ss_img1;
          std::stringstream ss_img2;
          ss_img1 << "visualize/" << img_key << "_" << n << "_conv_img1.jpg";
          ss_img2 << "visualize/" << img_key << "_" << n << "_conv_img2.jpg";
          cv::imwrite(ss_img1.str(), img1);
          cv::imwrite(ss_img2.str(), img2);
          LOG(INFO) << "Img:" << img_key << " wrote.";
        }
      }

    }

__global__ void sync_conv_groups() { }

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    /*TODO: Uncomment this to visualize


  std::stringstream ss;
  //		ss << "from_concat" << rand() % 1000;
  ss << bottom[0]->shape(3) << "_" << bottom[0]->shape(2) << "_tsn";

  visualize(bottom[0]->cpu_data(), bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(3),
            bottom[0]->shape(2), ss.str());
  */
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, weight + weight_offset_ * g,
            conv_descs_[i],
            fwd_algo_[i], workspaceData_fwd[g]->mutable_gpu_data(),
            workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
#if CUDNN_VERSION_MIN(4, 0, 0)
        CUDNN_CHECK(cudnnAddTensor(handle_[g],
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
#else
        CUDNN_CHECK(cudnnAddTensor(handle_[g], CUDNN_ADD_SAME_C,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
#endif
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
              handle_[1*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_filter_algo_[i], workspaceData_bwd_filter[g]->mutable_gpu_data(),
              workspace_bwd_filter_sizes_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + weight_offset_ * g));
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight + weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_data_algo_[i], workspaceData_bwd_data[g]->mutable_gpu_data(),
              workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
