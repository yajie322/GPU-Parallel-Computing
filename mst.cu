#include <stdio.h>
#include <limits.h>
#include "timerc.h"

#define N 1024

__global__ void parallelMST(int *graph, int *new_graph, int *edges, int *new_edges, int *roots, int *serial, int *result, int *n) {
	int tid = threadIdx.x;
	int size = *n; // the size of the current (shrinked) graph
	int original_size = size; // the size of the original graph

	// Initialize the result to be a graph with all vertices but no edges
	for (int i = 0; i < size; i++) {
		result[tid * size + i] = INT_MAX;
	}

	while (size > 1) { // While there are more than one trees to be merged
		// For each vertex, find the edge with minimum weight
		if (tid < size) {
			int dist = INT_MAX;
			for (int i = 0; i < size; i++) {
				if (graph[tid * size + i] < dist) { // if node tid is closer to node bid than previous nodes
					dist = graph[tid * size + i]; // record the shortest distance from node bid 
					roots[tid] = i; // record tid to be the new nearest neighbor
				}
			}

			// Mark the edge we found
			int a = edges[2*(tid * size + roots[tid])]; // get the first endpoint of chosen edge in the original graph
			int b = edges[2*(tid * size + roots[tid]) + 1]; // // get the second endpoint of chosen edge in the original graph
			result[a * original_size + b] = dist; // mark (a,b)
			result[b * original_size + a] = dist; // mark (b,a)
		}

		__syncthreads();


		// Find the super-vertex for each tree
		if (tid < size) {
			// calculate each node's root in the shrinked tree
			int root = roots[tid];
			while (roots[roots[root]] != root) {
				root = roots[root];
			}
			if (roots[root] < root) {
				root = roots[root];
			}
			roots[tid] = root;
		}
		__syncthreads();

		// Find the serial number of each grouped tree, i.e. 1, 2, 3, ....
		serial[tid] = -1;
		if (tid == 0) {
			int count = 0;
			for (int i = 0; i < size; i++) { // for each vertex
				if (serial[roots[i]] == -1) { // if its root has not yet been assigned a serial ID
					serial[roots[i]] = count; // then assign next serial number to it
					count++;
				}
			}
			*n = count; // update the size of the new graph to other threads
		}
		__syncthreads();

		// For each vertex, change its root to be the serial number assigned
		if (tid < size) {
			roots[tid] = serial[roots[tid]];
		}
		__syncthreads();
		
		int next_size = *n; // have each vertex agree on the new size

		// Initialize the new weight matrix
		if (tid < next_size) {
			for (int i = 0; i < next_size; i++) {
				new_graph[tid * next_size + i] = INT_MAX;
			}
		}
		
		__syncthreads();

		// Generate new weight matrix
		if (tid < size) {
			for (int i = 0; i < size; i++) { // for each node
				if (tid != i && roots[tid] != roots[i]) { // if we do not have same root
					if (graph[tid * size + i] < new_graph[roots[tid] * next_size + roots[i]]) {
						// if our distance is less than the current distance between our roots,
						// then update the new distance as our distance
						new_graph[roots[tid] * next_size + roots[i]] = graph[tid * size + i];
						new_graph[roots[i] * next_size + roots[tid]] = graph[tid * size + i];
						// record the original endpoints of our edge
						new_edges[2 * (roots[tid] * next_size + roots[i])] = edges[2 * (tid * size + i)];
						new_edges[2 * (roots[tid] * next_size + roots[i]) + 1] = edges[2 * (tid * size + i) + 1];
						new_edges[2 * (roots[i] * next_size + roots[tid])] = edges[2 * (tid * size + i)];
						new_edges[2 * (roots[i] * next_size + roots[tid]) + 1] = edges[2 * (tid * size + i) + 1];
					}
				}
			}
		}
		__syncthreads();

		size = next_size; // update the new size

		// update the graph and edge sets for next round
		if (tid < size) {
			for (int i = 0; i < size; i++) {
				graph[tid * size + i] = new_graph[tid * size + i];
				edges[2 * (tid * size + i)] = new_edges[2 * (tid * size + i)];
				edges[2 * (tid * size + i) + 1] = new_edges[2 * (tid * size + i) + 1];
			}
		}
		__syncthreads();
		
	}	
}

// returns the node with minimum edge
int minKey(int *key, int *mstSet, int size) {
	int min = INT_MAX;
	int minKey;
	for (int i = 0; i < size; i++) {
		if (mstSet[i] == 0 && key[i] < min) {
			min = key[i];
			minKey = i;
		}
	}
	return minKey;
}

int *sequentialMST(int *graph, int size) {
	int *mst = (int *) malloc(size * size * sizeof(int)); // To store final result MST
	int *mstSet = (int *) malloc(size * sizeof(int)); // Set of vertices that have not yet been included in the MST
	int *key = (int *) malloc(size * sizeof(int)); // Store the shorest edge for each vertex
	int *parent = (int *) malloc(size * sizeof(int)); // To record parent for each vertex

	// Intialization
	for (int i = 0; i < size; i++) {
		key[i] = INT_MAX;
		mstSet[i] = 0;
		for (int j = 0; j < size; j++) {
			mst[i * size + j] = INT_MAX;
		}
	}

	// First vertex is always picked first
	key[0] = 0;
	parent[0] = -1;

	for (int i = 0; i < size; i++) {
		int u = minKey(key, mstSet, size); // Find the vertex with minimum edge
		mstSet[u] = 1; // Mark the vertex as found

		// Include the vertex and weight into MST
		if (u != 0) {
			mst[u * size + parent[u]] = key[u];
			mst[parent[u] * size + u] = key[u];
		}

		// Update minimum edge for each neighbor of the chosen vertex
		for (int v = 0; v < size; v++) {
			int weight = graph[u * size + v];
			if (weight != INT_MAX && mstSet[v] == 0 && weight < key[v]) { // if vertex is not marked and needs a update
				parent[v] = u;
				key[v] = weight;
			}
		}
	}
	free(mstSet);
	free(key);
	free(parent);
	return mst;
}

int main() {
	int *graph = (int *) malloc(N * N * sizeof(int));
	int *edges = (int *) malloc(2 * N * N * sizeof(int));

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			graph[i * N + j] = INT_MAX;
		}
	}

	// graph[1] = 7; graph[2] = 4; graph[5] = 3;
	// graph[N] = 7; graph[N + 2] = 2;
	// graph[2 * N] = 4; graph[2 * N + 1] = 2; graph[2 * N + 3] = 1; graph[2 * N + 4] = 5;
	// graph[3 * N + 2] = 1; graph[3 * N + 4] = 6;
	// graph[4 * N + 2] = 5; graph[4 * N + 3] = 6;
	// graph[5 * N] = 3;

	// edges[2*5] = 0; edges[2*5+1] = 5; edges[2*(5*N)] = 0; edges[2*(5*N)+1] = 5;
	// edges[2*1] = 0; edges[2*1+1] = 1; edges[2*(1*N)] = 0; edges[2*(1*N)+1] = 1;
	// edges[2*2] = 0; edges[2*2+1] = 2; edges[2*(2*N)] = 0; edges[2*(2*N)+1] = 2;
	// edges[2*(1*N+2)] = 1; edges[2*(1*N+2)+1] = 2; edges[2*(2*N+1)] = 1; edges[2*(2*N+1)+1] = 2;
	// edges[2*(2*N+3)] = 2; edges[2*(2*N+3)+1] = 3; edges[2*(3*N+2)] = 2; edges[2*(3*N+2)+1] = 3;
	// edges[2*(2*N+4)] = 2; edges[2*(2*N+4)+1] = 4; edges[2*(4*N+2)] = 2; edges[2*(4*N+2)+1] = 4;
	// edges[2*(3*N+4)] = 3; edges[2*(3*N+4)+1] = 4; edges[2*(4*N+3)] = 3; edges[2*(4*N+3)+1] = 4;

	for (int i = 0; i < N; i++) {
		for (int j = i+1; j < N; j++) {
			int r = rand() % 100;
			if (r % 2) {
				graph[i * N + j] = r;
				graph[j * N + i] = r;
				edges[2*(i*N+j)] = i; edges[2*(i*N+j)+1] = j;
				edges[2*(j*N+i)] = i; edges[2*(j*N+i)+1] = j;
			}
		}
	}

	// CPU Test
	float ctime;
	cstart();
	int *mst = sequentialMST(graph, N);
	cend(&ctime);

	// for (int i = 0; i < N; i ++) {
	// 	for (int j = 0; j < N; j++) {
	// 		if (mst[i*N+j] == INT_MAX) {
	// 			printf(" %3d ", 0);
	// 		} else {
	// 			printf(" %3d ",mst[i * N + j]);
	// 		}
	// 	}
	// 	printf("\n");
	// }
	free(mst);
	printf("\n");
	printf("CPU time = %f\n", ctime);

	int *new_graph = (int *) malloc(N * N * sizeof(int));
	int *new_edges = (int *) malloc(2 * N * N * sizeof(int));
	int *roots = (int *) malloc(N * sizeof(int));
	int *serial = (int *) malloc(N * sizeof(int));
	int *results = (int *) malloc(N * N * sizeof(int));
	int n = N;

	int *graph_dev, *new_graph_dev, *edges_dev, *new_edges_dev, *roots_dev, *serial_dev, *results_dev, *size;
	cudaMalloc((void **) &graph_dev, N * N * sizeof(int));
	cudaMalloc((void **) &new_graph_dev, N * N * sizeof(int));
	cudaMalloc((void **) &edges_dev, 2 * N * N * sizeof(int));
	cudaMalloc((void **) &new_edges_dev, 2* N * N * sizeof(int));
	cudaMalloc((void **) &roots_dev, N * sizeof(int));
	cudaMalloc((void **) &serial_dev, N * sizeof(int));
	cudaMalloc((void **) &results_dev, N * N * sizeof(int));
	cudaMalloc((void **) &size, sizeof(int));

	float gtime_copy;
	gstart();
	cudaMemcpy(graph_dev, graph, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(new_graph_dev, new_graph, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(edges_dev, edges, 2 * N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(new_edges_dev, new_edges, 2 * N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(roots_dev, roots, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(serial_dev, serial, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(results_dev, results, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(size, &n, sizeof(int), cudaMemcpyHostToDevice);
	gend(&gtime_copy);
	printf("Time to copy input = %f\n", gtime_copy);

	float gtime;
	gstart();
	parallelMST<<<1, N>>>(graph_dev, new_graph_dev, edges_dev, new_edges_dev, roots_dev, serial_dev, results_dev, size);
	gend(&gtime);
	printf("GPU time = %f\n", gtime);

	float gtime_output;
	gstart();
	cudaMemcpy(results, results_dev, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	gend(&gtime_output);
	printf("Time to copy output = %f\n", gtime_output);
	// for (int i = 0; i < N; i ++) {
	// 	for (int j = 0; j < N; j++) {
	// 		if (results[i*N+j] == INT_MAX) {
	// 			printf(" %3d ", 0);
	// 		} else {
	// 			printf(" %3d ",results[i * N + j]);
	// 		}
	// 	}
	// 	printf("\n");
	// }

	cudaFree(graph_dev); cudaFree(new_graph_dev); cudaFree(roots_dev); cudaFree(serial_dev);
	cudaFree(results_dev); cudaFree(size);
	free(graph); free(new_graph); free(roots); free(serial); free(results);
}