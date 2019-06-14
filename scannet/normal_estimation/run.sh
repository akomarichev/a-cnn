#/usr/bin/sh
python 1_read_and_save_as_files.py
rm -rf build/
mkdir build
cd build
cmake ..
make
./2_compute_normals_and_curvature
python 3_pickle_normals.py