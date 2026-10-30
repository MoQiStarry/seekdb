#ifndef SEEKDB_CUVS_BRIDGE_H
#define SEEKDB_CUVS_BRIDGE_H
#ifdef __cplusplus
extern "C" {
#endif
/* One-shot: build a CAGRA index over base (n x dim, row-major f32) on the GPU and
 * search queries (nq x dim), writing top-k uint32 neighbor ids to out_ids
 * (length nq*topk). Returns 0 on success, non-zero on error. */
int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids);

/* Build-once / search-many (used by seekdb's ob_vsag_adaptor GPU path).
 *  - seekdb_cuvs_build: builds a CAGRA index on the GPU from base (n x dim,
 *    row-major f32) and returns an opaque handle (NULL on error). cuVS copies
 *    the dataset to the device, so `base` may be freed after this returns.
 *  - seekdb_cuvs_search: runs top-k search for nq queries (row-major f32, dim
 *    cols), writing uint32 row offsets to out_ids[nq*topk] and (if non-NULL) L2
 *    distances to out_dist[nq*topk]. Returns 0 on success.
 *  - seekdb_cuvs_free: releases the handle.
 * All ROCm/cuVS state lives inside the .so; seekdb only ever sees these C symbols. */
void* seekdb_cuvs_build(const float* base, long n, long dim);
int   seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                         unsigned int* out_ids, float* out_dist);
void  seekdb_cuvs_free(void* handle);
int seekdb_cuvs_serialize(void* handle, const char* filename);
void* seekdb_cuvs_deserialize(const char* filename, long n, long dim);

/* OBCUVS03 graph hand-off (keeps one copy of the vectors instead of two).
 *  - seekdb_cuvs_graph_shape: rows(== n) and cols(== graph_degree) of the row-major
 *    uint32 CAGRA graph the index holds on the GPU.
 *  - seekdb_cuvs_graph_bytes: copy `len` bytes of that graph from device offset
 *    `offset` into a host buffer, so the caller can stream it in chunks.
 *  - seekdb_cuvs_index_from_graph: rebuild a searchable GPU index from an exported
 *    graph plus the host dataset (n x dim, row-major f32). The index copies both
 *    tensors to the device and owns its copies. Returns NULL on error.
 * The dataset is NOT stored in the graph, so the graph is only meaningful together
 * with the exact same vector buffer: rows must equal n or the call fails. */
int   seekdb_cuvs_graph_shape(void* handle, long* rows, long* cols);
int   seekdb_cuvs_graph_bytes(void* handle, long offset, long len, void* dst);
void* seekdb_cuvs_index_from_graph(const unsigned int* graph, long rows, long cols,
                                   const float* dataset, long n, long dim);
#ifdef __cplusplus
}
#endif
#endif
