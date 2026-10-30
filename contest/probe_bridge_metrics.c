/* contest/probe_bridge_metrics.c — bridge 层的进程内显存与单查询延迟探针
 *
 * run_build_bench.sh 用 rocm-smi 采样的是**整卡**显存，共享实例上会被其他租户污染；
 * 本探针用 hipMemGetInfo 直接量本进程占用（需 GPU 空闲），并给 nq=1 的 bridge 调用
 * 计时，用来把端到端 QPS 拆成「bridge 时间」与「SQL 层固定开销」。
 *
 * 编译（评测机）：
 *   hipcc -O2 -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include probe_bridge_metrics.c \
 *     -Wl,-rpath,/opt/hipvs/lib:/opt/rocm/lib -ldl -o probe_bridge_metrics
 * 运行：
 *   ./probe_bridge_metrics <libseekdb_cuvs_bridge.so> <base.f32> <query.f32> <n> <dim> <iters>
 * 输出：
 *   free_mb start/after_build/after_queries 与 nq1_top10_us / nq1_top100_us
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <hip/hip_runtime.h>

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static float *loadf(const char *p, size_t cnt) {
  FILE *f = fopen(p, "rb");
  if (!f) { perror(p); exit(2); }
  float *b = (float *)malloc(cnt * 4);
  if (fread(b, 4, cnt, f) != cnt) { fprintf(stderr, "short read: %s\n", p); exit(2); }
  fclose(f);
  return b;
}

static double freemb(void) {
  size_t f = 0, t = 0;
  hipMemGetInfo(&f, &t);
  return f / 1048576.0;
}

int main(int argc, char **argv) {
  if (argc < 7) {
    fprintf(stderr, "usage: %s bridge.so base.f32 query.f32 n dim iters\n", argv[0]);
    return 2;
  }
  void *h = dlopen(argv[1], RTLD_NOW);
  if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
  void *(*bld)(const float *, long, long) = dlsym(h, "seekdb_cuvs_build");
  int (*srch)(void *, const float *, long, long, unsigned *, float *) =
      dlsym(h, "seekdb_cuvs_search");
  void (*fre)(void *) = dlsym(h, "seekdb_cuvs_free");
  if (!bld || !srch || !fre) { fprintf(stderr, "missing bridge symbols\n"); return 2; }

  long n = atol(argv[4]);
  int dim = atoi(argv[5]), iters = atoi(argv[6]);
  float *base = loadf(argv[2], (size_t)n * dim);
  float *q = loadf(argv[3], (size_t)dim);

  printf("free_mb start=%.1f\n", freemb());
  void *ih = bld(base, n, dim);
  if (!ih) { fprintf(stderr, "build failed\n"); return 3; }
  printf("free_mb after_build=%.1f\n", freemb());

  unsigned ids[100];
  float dst[100];
  for (int i = 0; i < 200; ++i) srch(ih, q, 1, 10, ids, dst);
  double t0 = now();
  for (int i = 0; i < iters; ++i) srch(ih, q, 1, 10, ids, dst);
  printf("nq1_top10_us=%.1f\n", (now() - t0) * 1e6 / iters);

  t0 = now();
  for (int i = 0; i < iters; ++i) srch(ih, q, 1, 100, ids, dst);
  printf("nq1_top100_us=%.1f\n", (now() - t0) * 1e6 / iters);
  printf("free_mb after_queries=%.1f\n", freemb());

  fre(ih);
  printf("free_mb after_free=%.1f\n", freemb());
  return 0;
}
