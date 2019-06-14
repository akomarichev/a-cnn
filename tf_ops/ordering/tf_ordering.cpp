/* The module orders neighbors in counterclockwise manner.
 * Author: Artem Komarichev
 * All Rights Reserved. 2018.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>

using namespace tensorflow;
using namespace std;

REGISTER_OP("OrderNeighbors")
  .Attr("k: int")
  .Input("input_xyz: float32")
  .Input("query_xyz: float32")
  .Input("query_normals: float32")
  .Input("idx: int32")
  .Output("outi: int32")
  .Output("proj: float32")
  .Output("angles: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
    int k;
    TF_RETURN_IF_ERROR(c->GetAttr("k", &k));
    ::tensorflow::shape_inference::ShapeHandle dims2;
    c->WithRank(c->input(3), 3, &dims2);
    ::tensorflow::shape_inference::ShapeHandle dims1;
    c->WithRank(c->input(0), 3, &dims1);
    ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), k, c->Dim(dims1,2)});
    c->set_output(0, c->input(3));
    c->set_output(1, output2);
    c->set_output(2, c->input(3));
    return Status::OK();
  });

void orderNeighborsLauncher(int b, int m, int n, int m_q, int k, const float *input, const float *queries, const float *queries_norm, const int *idx, float *proj, int *outi, float *angles);
class OrderNeighborsGpuOp : public OpKernel {
      public:
          explicit OrderNeighborsGpuOp(OpKernelConstruction * context):OpKernel(context){
            OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
            OP_REQUIRES(context, k_ > 0, errors::InvalidArgument("OrderNeighbors expects positive k"));
          }

          void Compute(OpKernelContext* context) override {
            const Tensor& input_xyz_tensor = context->input(0);
            OP_REQUIRES(context, input_xyz_tensor.dims() == 3, errors::InvalidArgument("OrderNeighbors expects (b,m,n) input_xyz shape"));

            int b = input_xyz_tensor.shape().dim_size(0);
            int m = input_xyz_tensor.shape().dim_size(1);
            int n = input_xyz_tensor.shape().dim_size(2);

            const Tensor& query_xyz_tensor = context->input(1);
            OP_REQUIRES(context, query_xyz_tensor.dims() == 3, errors::InvalidArgument("OrderNeighbors expects (b,m_q,n) query_xyz shape"));

            int m_q = query_xyz_tensor.shape().dim_size(1);

            const Tensor& query_normals_tensor = context->input(2);
            OP_REQUIRES(context, query_normals_tensor.dims() == 3, errors::InvalidArgument("OrderNeighbors expects (b,m_q,n) query_normals shape"));

            const Tensor& idx_tensor = context->input(3);
            OP_REQUIRES(context, idx_tensor.dims() == 3, errors::InvalidArgument("OrderNeighbors expects (b,m_q,k) idx shape"));

            int k = idx_tensor.shape().dim_size(2);

            Tensor *outi_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m_q,k}, &outi_tensor));
            Tensor *proj_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m_q,k,n}, &proj_tensor));
            Tensor *angles_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{b,m_q,k}, &angles_tensor));

            auto input_flat = input_xyz_tensor.flat<float>();
            const float *input = &(input_flat(0));
            auto queries_flat = query_xyz_tensor.flat<float>();
            const float *queries = &(queries_flat(0));
            auto queries_norm_flat = query_normals_tensor.flat<float>();
            const float *queries_norm = &(queries_norm_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto outi_flat = outi_tensor->flat<int>();
            int *outi = &(outi_flat(0));
            auto proj_flat = proj_tensor->flat<float>();
            float *proj = &(proj_flat(0));
            cudaMemset(proj, 0.0, sizeof(float)*b*m_q*k*n);
            auto angles_flat = angles_tensor->flat<float>();
            float *angles = &(angles_flat(0));
            cudaMemset(angles, 0.0, sizeof(float)*b*m_q*k);
            orderNeighborsLauncher(b, m, n, m_q, k, input, queries, queries_norm, idx, proj, outi, angles);
          }

        private:
          int k_;
};
REGISTER_KERNEL_BUILDER(Name("OrderNeighbors").Device(DEVICE_GPU), OrderNeighborsGpuOp);
