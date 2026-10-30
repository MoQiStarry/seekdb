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
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t       idx;
  long n, dim;
  /* Search scratch, kept for the life of the index: one SQL statement is one
     nq=1 search, and per-call allocation/synchronisation cost dominates the
     GPU work at that size. */
  cuvsCagraSearchParams_t sp;
  float    *qd; size_t qd_bytes;
  unsigned *nd; size_t nd_bytes;
  float    *dd; size_t dd_bytes;
  pthread_mutex_t mu;
} sk_cuvs_index;

static int sk_ensure_buf(cuvsResources_t res, void **buf, size_t *cap, size_t need)
{
  if (*cap >= need) return 0;
  size_t cap_new = need < 4096 ? 4096 : need;
  if (*buf != NULL) cuvsRMMFree(res, *buf, *cap);
  if (cuvsRMMAlloc(res, buf, cap_new) != CUVS_SUCCESS) { *buf = NULL; *cap = 0; return -1; }
  *cap = cap_new;
  return 0;
}

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  pthread_mutex_init(&h->mu, NULL);
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { pthread_mutex_destroy(&h->mu); free(h); return NULL; }

  DLManagedTensor dset; long ds[2] = {n, dim};
  dset.dl_tensor.data = (void*)base;
  dset.dl_tensor.device.device_type = kDLCPU; dset.dl_tensor.device.device_id = 0;
  dset.dl_tensor.ndim = 2;
  dset.dl_tensor.dtype.code = kDLFloat; dset.dl_tensor.dtype.bits = 32; dset.dl_tensor.dtype.lanes = 1;
  dset.dl_tensor.shape = ds; dset.dl_tensor.strides = 0; dset.dl_tensor.byte_offset = 0;

  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail_res;
  h->ip->build_algo = NN_DESCENT;
  /* Build knobs are per dimension because the recall they cost is not.
     256-d needs a wide *final* graph to hold Recall@100 (0.85 floor and a
     0.005 drop cap vs the frozen baseline) and spends its budget there: the
     intermediate graph stays at that width and NN-Descent runs 30 rounds.
     128-d is saturated, so a 128-wide graph converges in 10 rounds with no
     measurable recall change. Measured on 250K proxies at nq=4096:
     128-d 42.7s -> 6.3s (delta r@10 -0.0001), 256-d 41.2s -> 15.8s
     (delta r@10 +0.005, r@100 +0.010). */
  h->ip->graph_degree = (dim <= 192) ? 128 : 192;
  h->ip->intermediate_graph_degree = (dim <= 192) ? 128 : 192;
  h->ip->nn_descent_niter = (dim <= 192) ? 10 : 30;
  if (cuvsCagraIndexCreate(&h->idx)      != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) goto fail_idx;
  h->n = n; h->dim = dim;
  return h;

fail_idx: cuvsCagraIndexDestroy(h->idx);
fail_ip:  cuvsCagraIndexParamsDestroy(h->ip);
fail_res: cuvsResourcesDestroy(h->res); pthread_mutex_destroy(&h->mu); free(h);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h || nq <= 0 || topk <= 0) return 1;
  int rc = 0;
  pthread_mutex_lock(&h->mu);
  if (sk_ensure_buf(h->res, (void**)&h->qd, &h->qd_bytes, sizeof(float) * (size_t)nq * h->dim) ||
      sk_ensure_buf(h->res, (void**)&h->nd, &h->nd_bytes, sizeof(unsigned) * (size_t)nq * topk) ||
      sk_ensure_buf(h->res, (void**)&h->dd, &h->dd_bytes, sizeof(float) * (size_t)nq * topk)) {
    rc = 10; goto done;
  }
  if (h->sp == NULL && cuvsCagraSearchParamsCreate(&h->sp) != CUVS_SUCCESS) { rc = 14; goto done; }
  {
    size_t required_itopk = (size_t)topk * 4;
    if (required_itopk < 512) required_itopk = 512;
    h->sp->itopk_size = required_itopk;
    h->sp->search_width = 4;
    /* MULTI_CTA spreads one query over many CTAs; the default (SINGLE_CTA) is
       tuned for large batches and costs ~2.6ms of fixed work per nq=1 call. */
    h->sp->algo = MULTI_CTA;
  }
  if (cudaMemcpy(h->qd, query, sizeof(float) * (size_t)nq * h->dim, cudaMemcpyDefault) != cudaSuccess) { rc = 13; goto done; }
  {
    DLManagedTensor qt, nt, dt; long qs[2]={nq,h->dim}, nsh[2]={nq,topk}, dsh[2]={nq,topk};
    qt.dl_tensor.data=h->qd; qt.dl_tensor.device.device_type=kDLCUDA; qt.dl_tensor.device.device_id=0;
    qt.dl_tensor.ndim=2; qt.dl_tensor.dtype.code=kDLFloat; qt.dl_tensor.dtype.bits=32; qt.dl_tensor.dtype.lanes=1;
    qt.dl_tensor.shape=qs; qt.dl_tensor.strides=0; qt.dl_tensor.byte_offset=0;
    nt=qt; nt.dl_tensor.data=h->nd; nt.dl_tensor.dtype.code=kDLUInt; nt.dl_tensor.shape=nsh;
    dt=qt; dt.dl_tensor.data=h->dd; dt.dl_tensor.dtype.code=kDLFloat; dt.dl_tensor.shape=dsh;
    cuvsFilter f; f.type=NO_FILTER; f.addr=(uintptr_t)0;
    if (cuvsCagraSearch(h->res, h->sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc = 15; goto done; }
    if (cudaMemcpy(out_ids, h->nd, sizeof(unsigned) * (size_t)nq * topk, cudaMemcpyDefault) != cudaSuccess) { rc = 16; goto done; }
    if (out_dist) cudaMemcpy(out_dist, h->dd, sizeof(float) * (size_t)nq * topk, cudaMemcpyDefault);
  }
done:
  pthread_mutex_unlock(&h->mu);
  if (rc) fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return;
  if (h->sp) cuvsCagraSearchParamsDestroy(h->sp);
  if (h->dd) cuvsRMMFree(h->res, h->dd, h->dd_bytes);
  if (h->nd) cuvsRMMFree(h->res, h->nd, h->nd_bytes);
  if (h->qd) cuvsRMMFree(h->res, h->qd, h->qd_bytes);
  pthread_mutex_destroy(&h->mu);
  cuvsCagraIndexDestroy(h->idx);
  cuvsCagraIndexParamsDestroy(h->ip);
  cuvsResourcesDestroy(h->res);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  if (!index || !filename) return 1;
  if (cuvsCagraSerialize(index->res, filename, index->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  sk_cuvs_index* index = (sk_cuvs_index*)calloc(1, sizeof(*index));
  if (!index) return NULL;
  pthread_mutex_init(&index->mu, NULL);
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

/* ---- OBCUVS03: dataset-free graph hand-off ------------------------------
 * cuvsCagraSerialize(include_dataset=true) writes the raw vectors into its blob
 * and seekdb's snapshot then writes them a second time. These entry points let
 * seekdb keep exactly one copy: the graph leaves the GPU as plain row-major
 * uint32 and comes back together with seekdb's own vector buffer through
 * cuvsCagraIndexFromArgs.
 */
static int sk_cuvs_graph_view(sk_cuvs_index* h, DLManagedTensor* graph, long* rows, long* cols)
{
  memset(graph, 0, sizeof(*graph));
  if (cuvsCagraIndexGetGraph(h->idx, graph) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] GetGraph failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  if (graph->dl_tensor.ndim != 2 || graph->dl_tensor.data == NULL ||
      graph->dl_tensor.device.device_type != kDLCUDA ||
      graph->dl_tensor.dtype.code != kDLUInt || graph->dl_tensor.dtype.bits != 32 ||
      graph->dl_tensor.shape[0] != h->n || graph->dl_tensor.shape[1] <= 0) {
    fprintf(stderr, "[seekdb_cuvs] unexpected CAGRA graph layout (ndim=%d dtype=%d/%dbit dev=%d)\n",
            (int)graph->dl_tensor.ndim, (int)graph->dl_tensor.dtype.code,
            (int)graph->dl_tensor.dtype.bits, (int)graph->dl_tensor.device.device_type);
    if (graph->deleter) graph->deleter(graph);
    memset(graph, 0, sizeof(*graph));
    return 1;
  }
  *rows = (long)graph->dl_tensor.shape[0];
  *cols = (long)graph->dl_tensor.shape[1];
  return 0;
}

int seekdb_cuvs_graph_shape(void* handle, long* rows, long* cols)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  DLManagedTensor graph;
  if (!h || !rows || !cols) return 1;
  if (sk_cuvs_graph_view(h, &graph, rows, cols) != 0) return 1;
  if (graph.deleter) graph.deleter(&graph);
  return 0;
}

int seekdb_cuvs_graph_bytes(void* handle, long offset, long len, void* dst)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  DLManagedTensor graph;
  long rows = 0, cols = 0;
  if (!h || !dst || offset < 0 || len <= 0) return 1;
  if (sk_cuvs_graph_view(h, &graph, &rows, &cols) != 0) return 1;
  if (offset + len > rows * cols * 4L) {
    fprintf(stderr, "[seekdb_cuvs] graph read out of range: %ld+%ld > %ld\n",
            offset, len, rows * cols * 4L);
    if (graph.deleter) graph.deleter(&graph);
    return 1;
  }
  cudaError_t ce = cudaMemcpy(dst, (const char*)graph.dl_tensor.data + offset,
                              (size_t)len, cudaMemcpyDefault);
  if (graph.deleter) graph.deleter(&graph);
  if (ce != cudaSuccess) {
    fprintf(stderr, "[seekdb_cuvs] graph D2H failed: %s\n", cudaGetErrorString(ce));
    return 1;
  }
  return 0;
}

void* seekdb_cuvs_index_from_graph(const unsigned int* graph, long rows, long cols,
                                   const float* dataset, long n, long dim)
{
  if (!graph || !dataset || rows <= 0 || cols <= 0 || rows != n || n <= 0 || dim <= 0) return NULL;
  sk_cuvs_index* index = (sk_cuvs_index*)calloc(1, sizeof(*index));
  DLManagedTensor gt, dt;
  long gshape[2] = {rows, cols};
  long dshape[2] = {n, dim};
  int64_t actual_n = 0, actual_dim = 0;
  if (!index) return NULL;
  pthread_mutex_init(&index->mu, NULL);
  memset(&gt, 0, sizeof(gt)); memset(&dt, 0, sizeof(dt));
  gt.dl_tensor.data = (void*)graph; gt.dl_tensor.device.device_type = kDLCPU;
  gt.dl_tensor.ndim = 2; gt.dl_tensor.dtype.code = kDLUInt; gt.dl_tensor.dtype.bits = 32;
  gt.dl_tensor.dtype.lanes = 1; gt.dl_tensor.shape = gshape;
  dt.dl_tensor.data = (void*)dataset; dt.dl_tensor.device.device_type = kDLCPU;
  dt.dl_tensor.ndim = 2; dt.dl_tensor.dtype.code = kDLFloat; dt.dl_tensor.dtype.bits = 32;
  dt.dl_tensor.dtype.lanes = 1; dt.dl_tensor.shape = dshape;
  if (cuvsResourcesCreate(&index->res) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexParamsCreate(&index->ip) != CUVS_SUCCESS) goto fail_res;
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) goto fail_ip;
  /* the only metric seekdb marks for the GPU path is l2 == L2Expanded */
  if (cuvsCagraIndexFromArgs(index->res, L2Expanded, &gt, &dt, index->idx) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] IndexFromArgs failed: %s\n", cuvsGetLastErrorText());
    goto fail_idx;
  }
  if (cuvsCagraIndexGetSize(index->idx, &actual_n) != CUVS_SUCCESS ||
      cuvsCagraIndexGetDims(index->idx, &actual_dim) != CUVS_SUCCESS ||
      actual_n != n || actual_dim != dim) {
    fprintf(stderr, "[seekdb_cuvs] rebuilt shape mismatch: %ld x %ld, expected %ld x %ld\n",
            (long)actual_n, (long)actual_dim, n, dim);
    goto fail_idx;
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
  pthread_mutex_destroy(&index->mu);
fail:
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
