/* The module orders neighbors in counterclockwise manner.
 * Author: Artem Komarichev
 * All Rights Reserved. 2018.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>

__global__ void order_neighbors_gpu(int b, int m, int n, int m_q, int k, const float *input, const float *queries, const float *queries_norm, const int *idx, float *proj, int *outi, float *angles){
  int batch_index = blockIdx.x;
  queries+=m_q*n*batch_index;
  queries_norm+=m_q*n*batch_index;
  idx+=m_q*k*batch_index;
  angles+=m_q*k*batch_index;
  outi+=m_q*k*batch_index;
  input+=m*n*batch_index;
  proj+=m_q*k*n*batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  // copy indecies from idx to outi
  for (int i=index; i<m_q; i+=stride)
      for (int j=0; j<k; ++j)
          outi[i*k + j] = idx[i*k + j];

  for(int i=index; i<m_q; i+=stride){
    // 1. Project point P on plane represented by normal (n_x, n_q, n_z);
    for(int j=0; j<k; ++j){
      int pnt_idx = idx[i*k+j];
      float x_p = input[pnt_idx*n + 0];
      float y_p = input[pnt_idx*n + 1];
      float z_p = input[pnt_idx*n + 2];
      float n_x = queries_norm[i*n + 0];
      float n_y = queries_norm[i*n + 1];
      float n_z = queries_norm[i*n + 2];
      float x_q = queries[i*n + 0];
      float y_q = queries[i*n + 1];
      float z_q = queries[i*n + 2];
      // 1.1 find a distance to a plane
      float d = (x_p - x_q)*n_x + (y_p - y_q)*n_y + (z_p - z_q)*n_z;
      // 1.2 Calculate coordinates of projected point on a plane
      proj[(i*k+j)*n + 0] = x_p - d*n_x;
      proj[(i*k+j)*n + 1] = y_p - d*n_y;
      proj[(i*k+j)*n + 2] = z_p - d*n_z;
    }

    // 2. Calculate angles.
    float curvature_x = proj[i*k*n + 0];
    float curvature_y = proj[i*k*n + 1];
    float curvature_z = proj[i*k*n + 2];
    float n_x = queries_norm[i*n + 0];
    float n_y = queries_norm[i*n + 1];
    float n_z = queries_norm[i*n + 2];
    float x_q = queries[i*n + 0];
    float y_q = queries[i*n + 1];
    float z_q = queries[i*n + 2];
    float BCx = curvature_x - x_q;
    float BCy = curvature_y - y_q;
    float BCz = curvature_z - z_q;

    for(int j=1; j<k; ++j){
      float x_p = proj[(i*k+j)*n+0];
      float y_p = proj[(i*k+j)*n+1];
      float z_p = proj[(i*k+j)*n+2];

      float ACx = x_p - x_q;
      float ACy = y_p - y_q;
      float ACz = z_p - z_q;

      float cross_pr_x = ACy*BCz - ACz*BCy;
      float cross_pr_y = ACz*BCx - ACx*BCz;
      float cross_pr_z = ACx*BCy - ACy*BCx;
      float det = cross_pr_x*n_x + cross_pr_y*n_y + cross_pr_z*n_z;
      float cos_theta = (ACx*BCx + ACy*BCy + ACz * BCz)/(sqrtf((ACx*ACx + ACy*ACy + ACz*ACz) * (BCx*BCx + BCy*BCy + BCz*BCz)));
      if(det < 0){
        angles[i*k+j-1] = - cos_theta - 2;
      }
      else{
        angles[i*k+j-1] = cos_theta;
      }
    }

    // 3. Sort neighbors according to its angle values.
    for (int s=1;s<k;++s) {
        int max=s;

        for (int t=s+1;t<k;++t) {
            if (angles[i*k+t-1]>angles[i*k+max-1]) {
                max = t;
            }
        }

        if (max!=s) {
            // swap angles
            float tmp = angles[i*k+max-1];
            angles[i*k+max-1] = angles[i*k+s-1];
            angles[i*k+s-1] = tmp;

            // swap indecies
            int tmpi = outi[i*k+max];
            outi[i*k+max] = outi[i*k+s];
            outi[i*k+s] = tmpi;

            // swap projections as well
            for(int l=0;l<3;++l){
              float tmpl = proj[(i*k+max)*n+l];
              proj[(i*k+max)*n+l] = proj[(i*k+s)*n+l];
              proj[(i*k+s)*n+l] = tmpl;
            }
        }
    }
  }
}


void orderNeighborsLauncher(int b, int m, int n, int m_q, int k, const float *input, const float *queries, const float *queries_norm, const int *idx, float *proj, int *outi, float *angles){
  order_neighbors_gpu<<<b,256>>>(b,m,n,m_q,k,input,queries,queries_norm,idx,proj,outi,angles);
  cudaDeviceSynchronize();
}
