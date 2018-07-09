//
// Created by kevin on 2/19/17.
//
//


#ifndef CAFFE_STORE_HPP
#define CAFFE_STORE_HPP

#include <string>
#include <vector>
#include <boost/ptr_container/ptr_vector.hpp>
using namespace std;

class ImagePair {
public:
	ImagePair(const string &img1, const string &img2, const string &flow1, const string &flow2) : img1(img1),
																								  img2(img2),
																								  flow1(flow1),
																								  flow2(flow2) {}

	string img1;
	string img2;
	string flow1;
	string flow2;
};
#endif //CAFFE_STORE_HPP
