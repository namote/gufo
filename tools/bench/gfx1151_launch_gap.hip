// Per-dispatch launch overhead and persistent grid-barrier cost on gfx1151.
//
// Build: nix develop -c tools/bench/build.sh gfx1151_launch_gap
// Run:   /tmp/gfx1151_launch_gap
//        HIP_FORCE_DEV_KERNARG=1 /tmp/gfx1151_launch_gap   (kernarg placement)
//
// Part 1 times a chain of tiny dependent kernels (each reads the value the
// previous one wrote, so none can overlap) as a plain stream and as a
// captured hipGraph replay, with and without a side-stream fork/join per
// "layer" (the Flash-Next decode shape: ~30 kernels per layer, one fork),
// with 16-byte and 1 KiB kernel arguments, and with a one-wave grid and a
// 40 x 256 grid. The result is wall time per dispatch: launch gap plus the
// kernel's own (tiny) execution.
//
// Part 2 times a grid-wide barrier inside one persistent kernel (atomic
// arrival counter + generation word + __threadfence) at one, two and four
// blocks per CU, after checking that the whole grid can be co-resident.
//
// Every chain result is validated on the host (the final counter must equal
// the number of increments), so a dropped or reordered launch fails loudly.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_OK(expr)                                                    \
  do {                                                                  \
    const hipError_t err_ = (expr);                                     \
    if (err_ != hipSuccess) {                                           \
      std::fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(err_));                            \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

namespace {

constexpr int kChain = 1500;     // dependent dispatches per measured chain
constexpr int kPerLayer = 30;    // dispatches between side-stream forks
constexpr int kSideKernels = 2;  // kernels on the side branch per fork
constexpr int kReps = 20;        // timed repetitions (median reported)

struct SmallArgs {
  std::uint32_t* counter;
  std::uint32_t add;
  std::uint32_t pad;
};
static_assert(sizeof(SmallArgs) == 16);

struct LargeArgs {
  std::uint32_t* counter;
  std::uint32_t add;
  std::uint32_t pad[253];
};
static_assert(sizeof(LargeArgs) == 1024);

// One thread of the grid reads the previous value and writes value + add:
// the stream order is the only thing making the chain correct.
template<typename Args>
__global__ void Step(Args args) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    volatile std::uint32_t* c = args.counter;
    *c = *c + args.add;
  }
}

__global__ void SideStep(std::uint32_t* side) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    *side += 1;
}

struct Shape {
  const char* name;
  dim3 grid, block;
};

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

template<typename Args>
void Enqueue(hipStream_t main, hipStream_t side, hipEvent_t fork,
             hipEvent_t join, const Shape& shape, std::uint32_t* counter,
             std::uint32_t* side_counter, bool forked) {
  Args args{};
  args.counter = counter;
  args.add = 1;
  for (int i = 0; i < kChain; ++i) {
    if (forked && i % kPerLayer == 0) {
      HIP_OK(hipEventRecord(fork, main));
      HIP_OK(hipStreamWaitEvent(side, fork, 0));
      for (int j = 0; j < kSideKernels; ++j)
        hipLaunchKernelGGL(SideStep, dim3(1), dim3(32), 0, side, side_counter);
      HIP_OK(hipEventRecord(join, side));
    }
    hipLaunchKernelGGL(Step<Args>, shape.grid, shape.block, 0, main, args);
    if (forked && i % kPerLayer == kPerLayer / 2)
      HIP_OK(hipStreamWaitEvent(main, join, 0));
  }
  HIP_OK(hipGetLastError());
}

struct ChainResult {
  double stream_us;
  double graph_us;
};

template<typename Args>
ChainResult MeasureChain(const Shape& shape, bool forked) {
  hipStream_t main, side;
  HIP_OK(hipStreamCreateWithFlags(&main, hipStreamNonBlocking));
  HIP_OK(hipStreamCreateWithFlags(&side, hipStreamNonBlocking));
  hipEvent_t fork, join, t0, t1;
  HIP_OK(hipEventCreateWithFlags(&fork, hipEventDisableTiming));
  HIP_OK(hipEventCreateWithFlags(&join, hipEventDisableTiming));
  HIP_OK(hipEventCreate(&t0));
  HIP_OK(hipEventCreate(&t1));
  std::uint32_t *counter, *side_counter;
  HIP_OK(hipMalloc(&counter, sizeof(std::uint32_t)));
  HIP_OK(hipMalloc(&side_counter, sizeof(std::uint32_t)));
  const auto check = [&](std::uint32_t runs) {
    std::uint32_t value = 0;
    HIP_OK(hipMemcpy(&value, counter, sizeof(value), hipMemcpyDeviceToHost));
    if (value != runs * kChain) {
      std::fprintf(stderr, "chain %s: counter %u, expected %u\n", shape.name,
                   value, runs * kChain);
      std::exit(1);
    }
  };
  const auto timed = [&](auto&& body) {
    std::vector<double> samples;
    // On `main` itself: the null stream does not order a non-blocking one.
    HIP_OK(hipMemsetAsync(counter, 0, sizeof(std::uint32_t), main));
    HIP_OK(hipMemsetAsync(side_counter, 0, sizeof(std::uint32_t), main));
    body();  // warm-up
    HIP_OK(hipStreamSynchronize(main));
    for (int rep = 0; rep < kReps; ++rep) {
      HIP_OK(hipEventRecord(t0, main));
      body();
      HIP_OK(hipEventRecord(t1, main));
      HIP_OK(hipEventSynchronize(t1));
      float ms = 0.0F;
      HIP_OK(hipEventElapsedTime(&ms, t0, t1));
      samples.push_back(1000.0 * ms / kChain);
    }
    check(kReps + 1);
    if (forked) {
      std::uint32_t side_value = 0;
      HIP_OK(hipMemcpy(&side_value, side_counter, sizeof(side_value),
                       hipMemcpyDeviceToHost));
      const std::uint32_t forks = (kChain + kPerLayer - 1) / kPerLayer;
      if (side_value != (kReps + 1) * forks * kSideKernels) {
        std::fprintf(stderr, "chain %s: side counter %u, expected %u\n",
                     shape.name, side_value,
                     (kReps + 1) * forks * kSideKernels);
        std::exit(1);
      }
    }
    return Median(samples);
  };
  ChainResult result{};
  result.stream_us = timed([&] {
    Enqueue<Args>(main, side, fork, join, shape, counter, side_counter, forked);
  });
  hipGraph_t graph;
  HIP_OK(hipStreamBeginCapture(main, hipStreamCaptureModeThreadLocal));
  Enqueue<Args>(main, side, fork, join, shape, counter, side_counter, forked);
  HIP_OK(hipStreamEndCapture(main, &graph));
  hipGraphExec_t exec;
  HIP_OK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  result.graph_us = timed([&] { HIP_OK(hipGraphLaunch(exec, main)); });
  HIP_OK(hipGraphExecDestroy(exec));
  HIP_OK(hipGraphDestroy(graph));
  HIP_OK(hipFree(counter));
  HIP_OK(hipFree(side_counter));
  HIP_OK(hipEventDestroy(fork));
  HIP_OK(hipEventDestroy(join));
  HIP_OK(hipEventDestroy(t0));
  HIP_OK(hipEventDestroy(t1));
  HIP_OK(hipStreamDestroy(main));
  HIP_OK(hipStreamDestroy(side));
  return result;
}

// Grid-wide barrier: every block's thread 0 fences its writes, arrives on
// `count`; the last arrival resets it and bumps `generation`, which the
// others spin on. Each barrier also checks the previous phase's writes of
// every block, so a broken barrier is detected.
__global__ void PersistentBarriers(std::uint32_t* count,
                                   std::uint32_t* generation,
                                   std::uint32_t* slots, int barriers,
                                   std::uint32_t* errors) {
  for (int b = 0; b < barriers; ++b) {
    if (threadIdx.x == 0)
      __hip_atomic_store(slots + blockIdx.x, static_cast<std::uint32_t>(b + 1),
                         __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (threadIdx.x == 0) {
      const std::uint32_t gen = __hip_atomic_load(generation, __ATOMIC_RELAXED,
                                                  __HIP_MEMORY_SCOPE_AGENT);
      __threadfence();
      if (atomicAdd(count, 1U) == gridDim.x - 1) {
        __hip_atomic_store(count, 0U, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
        __threadfence();
        __hip_atomic_fetch_add(generation, 1U, __ATOMIC_RELEASE,
                               __HIP_MEMORY_SCOPE_AGENT);
      } else {
        // Bounded: a grid that is not co-resident reports instead of hanging.
        for (std::uint32_t spins = 0;
             __hip_atomic_load(generation, __ATOMIC_ACQUIRE,
                               __HIP_MEMORY_SCOPE_AGENT) == gen;
             ++spins) {
          if (spins == (1U << 26)) {
            atomicAdd(errors, 1U << 16);
            break;
          }
          __builtin_amdgcn_s_sleep(1);
        }
      }
      __threadfence();
    }
    __syncthreads();
    // A neighbour's slot must already hold this phase.
    if (threadIdx.x == 0) {
      const std::uint32_t other =
          __hip_atomic_load(slots + (blockIdx.x + 1) % gridDim.x,
                            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if (other < static_cast<std::uint32_t>(b + 1))
        atomicAdd(errors, 1U);
    }
    __syncthreads();
  }
}

double MeasureBarrier(int blocks, int threads, int barriers) {
  int per_cu = 0;
  HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_cu, PersistentBarriers, threads, 0));
  hipDeviceProp_t prop{};
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  if (per_cu * prop.multiProcessorCount < blocks) {
    std::printf("  %4d blocks x %3d: not co-resident (%d per CU), skipped\n",
                blocks, threads, per_cu);
    return -1.0;
  }
  std::uint32_t *count, *generation, *slots, *errors;
  HIP_OK(hipMalloc(&count, 4));
  HIP_OK(hipMalloc(&generation, 4));
  HIP_OK(hipMalloc(&slots, 4 * blocks));
  HIP_OK(hipMalloc(&errors, 4));
  hipEvent_t t0, t1;
  HIP_OK(hipEventCreate(&t0));
  HIP_OK(hipEventCreate(&t1));
  const auto run = [&](int n) {
    HIP_OK(hipMemset(count, 0, 4));
    HIP_OK(hipMemset(generation, 0, 4));
    HIP_OK(hipMemset(slots, 0, 4 * blocks));
    HIP_OK(hipEventRecord(t0, nullptr));
    // A cooperative launch refuses a grid that cannot be co-resident.
    void* args[] = {&count, &generation, &slots, &n, &errors};
    HIP_OK(hipLaunchCooperativeKernel(
        reinterpret_cast<const void*>(PersistentBarriers), dim3(blocks),
        dim3(threads), args, 0, nullptr));
    HIP_OK(hipEventRecord(t1, nullptr));
    HIP_OK(hipEventSynchronize(t1));
    float ms = 0.0F;
    HIP_OK(hipEventElapsedTime(&ms, t0, t1));
    return 1000.0 * ms;
  };
  HIP_OK(hipMemset(errors, 0, 4));
  run(16);  // warm-up
  std::vector<double> per_barrier;
  for (int rep = 0; rep < 7; ++rep) {
    // The difference of two barrier counts removes the launch itself.
    const double a = run(barriers), b = run(2 * barriers);
    per_barrier.push_back((b - a) / barriers);
  }
  std::uint32_t err = 0;
  HIP_OK(hipMemcpy(&err, errors, 4, hipMemcpyDeviceToHost));
  if (err != 0) {
    std::fprintf(stderr, "barrier %d x %d: %u ordering errors\n", blocks,
                 threads, err);
    std::exit(1);
  }
  HIP_OK(hipFree(count));
  HIP_OK(hipFree(generation));
  HIP_OK(hipFree(slots));
  HIP_OK(hipFree(errors));
  HIP_OK(hipEventDestroy(t0));
  HIP_OK(hipEventDestroy(t1));
  return Median(per_barrier);
}

}  // namespace

int main() {
  hipDeviceProp_t prop{};
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  const char* kernarg = std::getenv("HIP_FORCE_DEV_KERNARG");
  std::printf("device %s, %d CUs, HIP_FORCE_DEV_KERNARG=%s\n", prop.gcnArchName,
              prop.multiProcessorCount,
              kernarg != nullptr ? kernarg : "(unset)");
  std::printf("\nPart 1: %d dependent dispatches, median of %d, us/dispatch\n",
              kChain, kReps);
  std::printf("%-10s %-6s %-7s %10s %10s\n", "grid", "args", "fork", "stream",
              "graph");
  const Shape shapes[] = {{"1x32", dim3(1), dim3(32)},
                          {"40x256", dim3(40), dim3(256)}};
  for (const auto& shape : shapes) {
    for (const bool forked : {false, true}) {
      const auto small = MeasureChain<SmallArgs>(shape, forked);
      std::printf("%-10s %-6s %-7s %10.2f %10.2f\n", shape.name, "16B",
                  forked ? "yes" : "no", small.stream_us, small.graph_us);
      const auto large = MeasureChain<LargeArgs>(shape, forked);
      std::printf("%-10s %-6s %-7s %10.2f %10.2f\n", shape.name, "1KiB",
                  forked ? "yes" : "no", large.stream_us, large.graph_us);
    }
  }
  std::printf("\nPart 2: persistent grid barrier, us/barrier (median of 7)\n");
  const int cus = prop.multiProcessorCount;
  for (const int per_cu : {1, 2, 4}) {
    for (const int threads : {64, 256}) {
      const double us = MeasureBarrier(cus * per_cu, threads, 2000);
      if (us >= 0.0)
        std::printf("  %4d blocks x %3d (%d/CU): %8.2f us\n", cus * per_cu,
                    threads, per_cu, us);
    }
  }
  return 0;
}
