#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
//#include <pcl/visualization/pcl_visualizer.h>
#include <boost/filesystem.hpp>

using namespace std;
namespace fs = boost::filesystem;

typedef pcl::PointCloud<pcl::PointXYZI> CloudXYZI;
typedef pcl::PointCloud<pcl::PointXYZRGBA> CloudXYZRGBA;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointNCloudT;

void load_scannet_list(const string& input_dir, vector<string>& input_fnames);
void read_scannet_pc(const fs::path& fileName, CloudXYZRGBA& cloud);
void write_scannet_supvoxel(pcl::PointCloud<pcl::PointXYZL>& output_cloud, const fs::path& srcpath);
#endif