#include <cstdio>
#include <float.h>

__global__ void ring_point_gpu(int b, int n, int m, float radius_in, float radius_out, int nsample, const float *xyz1, const float *xyz2, const int *idx2, int *idx, int *pts_cnt) {
    int batch_index = blockIdx.x;
    xyz1 += n*3*batch_index;
    xyz2 += m*3*batch_index;
    idx2 += m*batch_index;
    idx += m*nsample*batch_index;
    pts_cnt += m*batch_index; // counting how many unique points selected in the local region

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int j=index;j<m;j+=stride) {
        int cnt = 0;
        // create two variables to track farthest nearest point. Its index and distance
        int farthest_ind = -1;
        float farthest_dist = FLT_MIN;
        float * dist = new float[nsample];

        for (int k=0;k<n;++k) {
            float x2=xyz2[j*3+0];
            float y2=xyz2[j*3+1];
            float z2=xyz2[j*3+2];
            float x1=xyz1[k*3+0];
            float y1=xyz1[k*3+1];
            float z1=xyz1[k*3+2];
    	    float d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
            if (d >= radius_in && d<radius_out && cnt < nsample) {
                if (cnt==0) {
                    for (int l=0;l<nsample;++l){
                        idx[j*nsample+l] = k;
                        dist[l] = d;
                    }
                }
                if(d > farthest_dist){
                  farthest_ind = cnt;
                  farthest_dist = d;
                }
                idx[j*nsample+cnt] = k;
                dist[cnt] = d;
                cnt+=1;
            }
            else if(d < farthest_dist && d >= radius_in){  // && d > 1e-5     // found closer point than fatherst one found so far
              idx[j*nsample+farthest_ind] = k;
              dist[farthest_ind] = d;

              // update farthest distance and farthest index
              farthest_dist = FLT_MIN;
              farthest_ind = -1;
              for(int l=0; l<nsample; ++l){
                  if(dist[l] > farthest_dist){
                    farthest_ind = l;
                    farthest_dist = dist[l];
                  }
                }
            }
        }

        if(cnt==0){
          for (int l=0;l<nsample;++l){
              idx[j*nsample+l] = idx2[j];
          }
          cnt = 1;
        }
        else{
          // replace first found point with the furthest one.
          int temp = idx[j*nsample+0];
          idx[j*nsample+0] = idx[j*nsample+farthest_ind];
          idx[j*nsample+farthest_ind] = temp;
        }

        pts_cnt[j] = cnt;
        delete dist;
    }
}

// input: points (b,n,c), idx (b,m,nsample)
// output: out (b,m,nsample,c)
__global__ void group_point_gpu(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out) {
    int batch_index = blockIdx.x;
    points += n*c*batch_index;
    idx += m*nsample*batch_index;
    out += m*nsample*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                out[j*nsample*c+k*c+l] = points[ii*c+l];
            }
        }
    }
}

// input: grad_out (b,m,nsample,c), idx (b,m,nsample),
// output: grad_points (b,n,c)
__global__ void group_point_grad_gpu(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points) {
    int batch_index = blockIdx.x;
    idx += m*nsample*batch_index;
    grad_out += m*nsample*c*batch_index;
    grad_points += n*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                 atomicAdd(&grad_points[ii*c+l], grad_out[j*nsample*c+k*c+l]);
            }
        }
    }
}


void ringPointLauncher(int b, int n, int m, float radius_in, float radius_out, int nsample, const float *xyz1, const float *xyz2, const int *idx2, int *idx, int *pts_cnt) {
    ring_point_gpu<<<b,256>>>(b,n,m,radius_in,radius_out,nsample,xyz1,xyz2,idx2,idx,pts_cnt);
    //cudaDeviceSynchronize();
}
void groupPointLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out){
    group_point_gpu<<<b,256>>>(b,n,c,m,nsample,points,idx,out);
    //cudaDeviceSynchronize();
}
void groupPointGradLauncher(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points){
    group_point_grad_gpu<<<b,256>>>(b,n,c,m,nsample,grad_out,idx,grad_points);
    //cudaDeviceSynchronize();
}
