#include <cnpy.h>

cnpy::NpyArray npydata = cnpy::npy_load("/home/Guest/tb5zhh/datasets/ScanNet/scans/scene0000_00/scene0000_00_labels.npy");
double* data;

int main {
    data = npydata.data<double>();
    std::cout << data;
    return;
}
