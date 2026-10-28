/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here. */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t       idx;
  long n, dim;
  void* fp16_data;       // FP16 dataset on GPU (if using mixed precision)
  int use_fp16;          // whether this index uses FP16 internally
} sk_cuvs_index;

// FP32 to FP16 conversion (IEEE 754 half-precision)
static unsigned short float_to_half(float f) {
  unsigned int x = *((unsigned int*)&f);
  unsigned int sign = (x >> 16) & 0x8000;
  int exp = ((x >> 23) & 0xff) - 127 + 15;
  unsigned int mantissa = x & 0x7fffff;
  
  if (exp <= 0) {
    return (unsigned short)sign;
  } else if (exp >= 31) {
    return (unsigned short)(sign | 0x7c00);
  }
  return (unsigned short)(sign | (exp << 10) | (mantissa >> 13));
}

static void convert_f32_to_f16(const float* src, unsigned short* dst, long count) {
  for (long i = 0; i < count; i++) {
    dst[i] = float_to_half(src[i]);
  }
}

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h); return NULL; }

  // Convert FP32 to FP16 for faster build (mixed precision)
  long total = n * dim;
  unsigned short* fp16_host = (unsigned short*)malloc(sizeof(unsigned short) * total);
  if (!fp16_host) goto fail_res;
  convert_f32_to_f16(base, fp16_host, total);
  
  // Allocate FP16 buffer on GPU
  void* fp16_gpu = NULL;
  if (cuvsRMMAlloc(h->res, &fp16_gpu, sizeof(unsigned short) * total) != CUVS_SUCCESS) {
    free(fp16_host);
    goto fail_res;
  }
  if (cudaMemcpy(fp16_gpu, fp16_host, sizeof(unsigned short) * total, cudaMemcpyDefault) != cudaSuccess) {
    cuvsRMMFree(h->res, fp16_gpu, sizeof(unsigned short) * total);
    free(fp16_host);
    goto fail_res;
  }
  free(fp16_host);
  
  h->fp16_data = fp16_gpu;
  h->use_fp16 = 1;

  DLManagedTensor dset; long ds[2] = {n, dim};
  dset.dl_tensor.data = fp16_gpu;
  dset.dl_tensor.device.device_type = kDLCUDA; dset.dl_tensor.device.device_id = 0;
  dset.dl_tensor.ndim = 2;
  dset.dl_tensor.dtype.code = kDLFloat; dset.dl_tensor.dtype.bits = 16; dset.dl_tensor.dtype.lanes = 1;
  dset.dl_tensor.shape = ds; dset.dl_tensor.strides = 0; dset.dl_tensor.byte_offset = 0;

  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail_fp16;
  h->ip->build_algo = NN_DESCENT;
  h->ip->graph_degree = 128;
  h->ip->intermediate_graph_degree = 256;
  h->ip->nn_descent_niter = 100;
  if (cuvsCagraIndexCreate(&h->idx)      != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) goto fail_idx;
  h->n = n; h->dim = dim;
  return h;

fail_idx: cuvsCagraIndexDestroy(h->idx);
fail_ip:  cuvsCagraIndexParamsDestroy(h->ip);
fail_fp16: cuvsRMMFree(h->res, fp16_gpu, sizeof(unsigned short) * total);
fail_res: cuvsResourcesDestroy(h->res); free(h);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return 1;
  int rc = 0;
  void* qd = 0; unsigned* nd = 0; float* dd = 0; cuvsCagraSearchParams_t sp = 0;
  
  // Allocate query buffer on GPU (FP16 if index uses FP16, else FP32)
  size_t query_elem_size = h->use_fp16 ? sizeof(unsigned short) : sizeof(float);
  if (cuvsRMMAlloc(h->res, &qd, query_elem_size * nq * h->dim) != CUVS_SUCCESS) { rc=10; goto done; }
  if (cuvsRMMAlloc(h->res, &nd, sizeof(unsigned) * nq * topk) != CUVS_SUCCESS) { rc=11; goto done; }
  if (cuvsRMMAlloc(h->res, &dd, sizeof(float) * nq * topk) != CUVS_SUCCESS) { rc=12; goto done; }
  
  // Convert and transfer query
  if (h->use_fp16) {
    // Convert FP32 query to FP16 on host, then transfer
    unsigned short* fp16_query = (unsigned short*)malloc(sizeof(unsigned short) * nq * h->dim);
    if (!fp16_query) { rc=13; goto done; }
    convert_f32_to_f16(query, fp16_query, nq * h->dim);
    if (cudaMemcpy(qd, fp16_query, sizeof(unsigned short) * nq * h->dim, cudaMemcpyDefault) != cudaSuccess) {
      free(fp16_query); rc=13; goto done;
    }
    free(fp16_query);
  } else {
    if (cudaMemcpy(qd, query, sizeof(float) * nq * h->dim, cudaMemcpyDefault) != cudaSuccess) { rc=13; goto done; }
  }
  
  {
    DLManagedTensor qt, nt, dt; long qs[2]={nq,h->dim}, nsh[2]={nq,topk}, dsh[2]={nq,topk};
    qt.dl_tensor.data=qd; qt.dl_tensor.device.device_type=kDLCUDA; qt.dl_tensor.device.device_id=0;
    qt.dl_tensor.ndim=2;
    qt.dl_tensor.dtype.code=kDLFloat; qt.dl_tensor.dtype.bits=h->use_fp16 ? 16 : 32; qt.dl_tensor.dtype.lanes=1;
    qt.dl_tensor.shape=qs; qt.dl_tensor.strides=0; qt.dl_tensor.byte_offset=0;
    nt=qt; nt.dl_tensor.data=nd; nt.dl_tensor.dtype.code=kDLUInt; nt.dl_tensor.shape=nsh;
    dt=qt; dt.dl_tensor.data=dd; dt.dl_tensor.dtype.code=kDLFloat; dt.dl_tensor.dtype.bits=32; dt.dl_tensor.shape=dsh;
    if (cuvsCagraSearchParamsCreate(&sp) != CUVS_SUCCESS) { rc=14; goto done; }
    {
      size_t required_itopk = (size_t)topk * 4;
      if (required_itopk < 512) required_itopk = 512;
      if (sp->itopk_size < required_itopk) sp->itopk_size = required_itopk;
      sp->search_width = 4;
    }
    cuvsFilter f; f.type=NO_FILTER; f.addr=(uintptr_t)0;
    if (cuvsCagraSearch(h->res, sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc=15; goto done; }
    if (cudaMemcpy(out_ids, nd, sizeof(unsigned)*nq*topk, cudaMemcpyDefault) != cudaSuccess) { rc=16; goto done; }
    if (out_dist) cudaMemcpy(out_dist, dd, sizeof(float)*nq*topk, cudaMemcpyDefault);
  }
done:
  if (sp) cuvsCagraSearchParamsDestroy(sp);
  if (dd) cuvsRMMFree(h->res, dd, sizeof(float)*nq*topk);
  if (nd) cuvsRMMFree(h->res, nd, sizeof(unsigned)*nq*topk);
  if (qd) cuvsRMMFree(h->res, qd, query_elem_size*nq*h->dim);
  if (rc) fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return;
  if (h->fp16_data) {
    cuvsRMMFree(h->res, h->fp16_data, sizeof(unsigned short) * h->n * h->dim);
  }
  cuvsCagraIndexDestroy(h->idx);
  cuvsCagraIndexParamsDestroy(h->ip);
  cuvsResourcesDestroy(h->res);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  if (!index || !filename) return 1;
  // Serialize the CAGRA index structure
  if (cuvsCagraSerialize(index->res, filename, index->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  // Note: FP16 dataset is not serialized with the index; it will be rebuilt on deserialize
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  sk_cuvs_index* index = (sk_cuvs_index*)calloc(1, sizeof(*index));
  if (!index) return NULL;
  index->fp16_data = NULL;
  index->use_fp16 = 0;
  if (cuvsResourcesCreate(&index->res) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexParamsCreate(&index->ip) != CUVS_SUCCESS) goto fail_res;
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraDeserialize(index->res, filename, index->idx) != CUVS_SUCCESS) goto fail_idx;
  {
    int64_t actual_n = 0;
    int64_t actual_dim = 0;
    if (cuvsCagraIndexGetSize(index->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(index->idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, n, dim);
      goto fail_idx;
    }
  }
  index->n = n;
  index->dim = dim;
  return index;
fail_idx:
  cuvsCagraIndexDestroy(index->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(index->ip);
fail_res:
  cuvsResourcesDestroy(index->res);
fail:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  free(index);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
