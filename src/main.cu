#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <set>

#include "timer.cpp"

using namespace std;

typedef uint64_t u64;
typedef int64_t i64;

#define GPU_ERROR_CHECK(x) { gpuAssert((x), __FILE__, __LINE__); }
#define _cudaMalloc(...) GPU_ERROR_CHECK(cudaMalloc(__VA_ARGS__));
#define _cudaMemcpy(...) GPU_ERROR_CHECK(cudaMemcpy(__VA_ARGS__));
#define _cudaDeviceSynchronize(...) GPU_ERROR_CHECK(cudaDeviceSynchronize(__VA_ARGS__));

__host__ inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code == cudaSuccess) return;
	fprintf(stderr, "GPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(code);
}

class AdjacencyGraph {
	public:
		u64 n;
		u64 m;
		u64* offsets;
		u64* edges;
		AdjacencyGraph(u64 n, u64 m) : n(n), m(m), offsets{new u64[n]}, edges{new u64[m]} {}
};

__host__ AdjacencyGraph load_adjacency_graph(const string& path) {
	auto file = ifstream(path);

	string t;
	u64 n, m;

	file >> t >> n >> m;

	assert(t == "AdjacencyGraph");

	auto graph = AdjacencyGraph(n, m);

	for (auto i = 0; i < n; i++) file >> graph.offsets[i];
	for (auto i = 0; i < m; i++) file >> graph.edges[i];

	return graph;
}

class BFSJob {
	public:
	// Generic
	bool* frontier;
	bool xdone = false;
	bool done = false;
	// BFS-Specific
	u64 root;
	i64* parents;

	BFSJob(u64 root) : root(root) {}

	__host__ void init(AdjacencyGraph& graph) {
		bool* h_frontier = new bool[graph.n];
		i64* h_parents = new i64[graph.n];
		for (u64 i = 0; i < graph.n; i++) {
			h_frontier[i] = i == root;
			h_parents[i] = i == root ? (i64) root : (i64) -1;
		}

		bool* d_frontier;
		i64* d_parents;
		auto x = graph.n * sizeof(bool);
		auto y = graph.n * sizeof(i64);
		_cudaMalloc(&d_frontier, x);
		_cudaMemcpy(d_frontier, h_frontier, x, cudaMemcpyHostToDevice);
		_cudaMalloc(&d_parents, y);
		_cudaMemcpy(d_parents, h_parents, y, cudaMemcpyHostToDevice);

		frontier = d_frontier;
		parents = d_parents;
	}

	__device__ void iter(u64* d_offsets, u64* d_edges, u64 n, u64 m, u64 id) {
		u64 a = d_offsets[id];
		u64 b = id < n - 1 ? d_offsets[id + 1] : n;

		frontier[id] = false;

		for (auto i = a; i < b; i++) {
			u64 c = d_edges[i];
			if (parents[c] != -1) continue;
			parents[c] = id;
			frontier[c] = true;
			done = false;
		}
	}
};

class SSSPJob {
	public:
	u64 source;
	i64* dists;

	SSSPJob(u64 source) : source(source) {}

	__host__ void init(AdjacencyGraph& graph) {
		i64* h_dists = new i64[graph.n];
		for (u64 i = 0; i < graph.n; i++) {
			h_dists[i] = i == source ? (i64) 0 : (i64) INT_FAST64_MAX;
		}

		i64* d_dists;
		auto y = graph.n * sizeof(i64);
		_cudaMalloc(&d_dists, y);
		_cudaMemcpy(d_dists, h_dists, y, cudaMemcpyHostToDevice);

		dists = d_dists;
	}

	__device__ void iter(u64* d_offsets, u64* d_edges, u64 n, u64 m, u64 id) {
		u64 a = d_offsets[id];
		u64 b = id < n - 1 ? d_offsets[id + 1] : n;

		for (auto i = a; i < b; i++) {
			auto dst = d_edges[i];
			if (dists[id] + 1 < dists[dst]) dists[dst] = dists[id] + 1;
		}
	}
};

__global__ void iter_bfs(u64* d_offsets, u64* d_edges, u64 n, u64 m, BFSJob* jobs, u64 j) {
	// Get our global thread ID
	u64 id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= n) return;

	for (auto i = 0; i < j; i++) {
		if (jobs[i].xdone) continue;
		if (!jobs[i].frontier[id]) continue;
		jobs[i].iter(d_offsets, d_edges, n, m, id);
	}
}

__global__ void iter_sssp(u64* d_offsets, u64* d_edges, u64 n, u64 m, SSSPJob* jobs, u64 j) {
	// Get our global thread ID
	u64 id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= n) return;

	for (auto i = 0; i < j; i++) {
		jobs[i].iter(d_offsets, d_edges, n, m, id);
	}
}

void exec_bfs(int count, AdjacencyGraph& graph, u64* d_offsets, u64* d_edges, int offset) {
	BFSJob* h_jobs = (BFSJob*) malloc(count * sizeof(BFSJob));
	BFSJob* d_jobs;
	for (auto i = 0; i < count; i++) {
		auto bfs = BFSJob((i + 1 + offset) * 10);
		bfs.init(graph);
		h_jobs[i] = bfs;
	}
	_cudaMalloc(&d_jobs, count * sizeof(BFSJob));
	_cudaMemcpy(d_jobs, h_jobs, count * sizeof(BFSJob), cudaMemcpyHostToDevice);

	// Number of threads in each thread block
	u64 blockSize = 1024;
 
	// Number of thread blocks in grid
	u64 gridSize = (u64) ceil((float) graph.n / blockSize);

	while (true) {
		auto done = true;
		bool* j_done = (bool*) malloc(sizeof(bool));
		for (auto i = 0; i < count; i++) {
			_cudaMemcpy(j_done, &(d_jobs[i].done), sizeof(bool), cudaMemcpyDeviceToHost);
			if (!(*j_done)) {
				done = false;
				*j_done = true;
				_cudaMemcpy(&(d_jobs[i].done), j_done, sizeof(bool), cudaMemcpyHostToDevice);
			} else {
				*j_done = true;
				_cudaMemcpy(&(d_jobs[i].xdone), j_done, sizeof(bool), cudaMemcpyHostToDevice);
			}
		}
		if (done) break;
		iter_bfs<<<gridSize, blockSize>>>(d_offsets, d_edges, graph.m, graph.n, d_jobs, count);
		_cudaDeviceSynchronize();
	}
}

void exec_sssp(int count, AdjacencyGraph& graph, u64* d_offsets, u64* d_edges, int offset) {
	SSSPJob* h_jobs = (SSSPJob*) malloc(count * sizeof(SSSPJob));
	SSSPJob* d_jobs;
	for (auto i = 0; i < count; i++) {
		auto sssp = SSSPJob((i + 1 + offset) * 10);
		sssp.init(graph);
		h_jobs[i] = sssp;
	}
	_cudaMalloc(&d_jobs, count * sizeof(SSSPJob));
	_cudaMemcpy(d_jobs, h_jobs, count * sizeof(SSSPJob), cudaMemcpyHostToDevice);

	// Number of threads in each thread block
	u64 blockSize = 1024;
 
	// Number of thread blocks in grid
	u64 gridSize = (u64) ceil((float) graph.n / blockSize);

	for (auto i = 0; i < 50; i++) {
		iter_sssp<<<gridSize, blockSize>>>(d_offsets, d_edges, graph.m, graph.n, d_jobs, count);
		_cudaDeviceSynchronize();
	}
}

__host__ int main(int argc, char **argv) {
	auto graph_path = string(argv[1]);
	auto job_count = (u64) atoi(argv[2]);
	auto run_bfs = (atoi(argv[3]) != 0);

	cout << "Graph Path: " << graph_path << endl;
	cout << "Job Count: " << job_count << endl;
	cout << "Job Type: " << (run_bfs ? "BFS" : "SSSP") << endl;

	auto host_io_time = custom::Timer("Host IO");

	auto graph = load_adjacency_graph(graph_path);
	
	host_io_time.report();

	auto device_io_time = custom::Timer("Device IO");

	// TODO: Create type for offset and another type for edge
	u64* d_offsets;
	u64* d_edges;

	auto offsets_size = graph.n * sizeof(u64);
	auto edges_size = graph.m * sizeof(u64);
 
	// Allocate memory for each vector on GPU
	_cudaMalloc(&d_offsets, offsets_size);
	_cudaMalloc(&d_edges, edges_size);

	// Copy host vectors to device
	_cudaMemcpy(d_offsets, graph.offsets, offsets_size, cudaMemcpyHostToDevice);
	_cudaMemcpy(d_edges, graph.edges, edges_size, cudaMemcpyHostToDevice);

	device_io_time.report();

	auto separated_jobs_time = custom::Timer("Running separated jobs");
	for (auto i = 0; i < job_count; i++) {
		if (run_bfs) exec_bfs(1, graph, d_offsets, d_edges, i);
		else exec_sssp(1, graph, d_offsets, d_edges, i);
	}
	separated_jobs_time.report();

	cout << "Running one job took " << (separated_jobs_time.seconds() / job_count) << " seconds." << std::endl;

	auto merged_jobs_time = custom::Timer("Running merged jobs");
	if (run_bfs) exec_bfs(job_count, graph, d_offsets, d_edges, 0);
	else exec_sssp(job_count, graph, d_offsets, d_edges, 0);
	merged_jobs_time.report();

	auto sep = separated_jobs_time.seconds();
	auto one = (sep / job_count);
	auto mer = merged_jobs_time.seconds();

	cout << one << "," << sep << "," << mer << endl;
}
