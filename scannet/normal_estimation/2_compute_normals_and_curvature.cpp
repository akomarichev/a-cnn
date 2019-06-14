#include <iostream>
#include <dirent.h>
#include <string>
#include <vector>
#include <fstream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>

using namespace std;

int main (int, char** argv)
{
  DIR * dir;
  struct dirent *pdir;
  vector<string> folders {"../../scannet_train/", "../../scannet_test/"};

  for(const auto path : folders){
    if((dir = opendir(path.c_str())) != NULL){
      while((pdir = readdir(dir)) != NULL){
        string fileName = pdir->d_name;
        if(fileName != "." && fileName != ".."){
          cout << "filename: " << fileName << endl;

          pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
          if (pcl::io::loadPCDFile<pcl::PointXYZ> (path+fileName, *cloud) == -1) //* load the file
          {
            PCL_ERROR ("Couldn't read file");
            return (-1);
          }

          // Compute the normals
          pcl::PointCloud<pcl::Normal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::Normal>);

          pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
          normal_estimation.setInputCloud (cloud);

          pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
          normal_estimation.setSearchMethod (tree);

          normal_estimation.setKSearch(10);

          normal_estimation.compute (*cloud_with_normals);

          std::ofstream myfile;
          size_t pos = fileName.find("_xyz.pcd");
          string outputFileName = path + fileName.substr(0, pos) + "_normals.txt";
          cout << "outputFileName: " + outputFileName << endl;
          myfile.open(outputFileName);

          for (size_t i = 0; i < cloud->points.size (); ++i)
                        myfile << cloud_with_normals->points[i].normal_x
                    << " "    << cloud_with_normals->points[i].normal_y
                    << " "    << cloud_with_normals->points[i].normal_z
                    << "\n";

          myfile.close();

          cout << "File " << outputFileName << " saved." << endl;
        }
      }
      closedir (dir);
    } else {
      perror("Could not open director.");
      return EXIT_FAILURE;
    }
  }
  return 0;
}
