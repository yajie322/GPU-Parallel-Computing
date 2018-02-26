#include <stdio.h>
#include <stdlib.h>
#include "timerc.h"

#define GAMEDIM 1024
#define GAMESIZE GAMEDIM * GAMEDIM

#define gerror(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

texture<int> game1_tex;
texture<int> game2_tex;

texture<int, 2> game1_tex2D;
texture<int, 2> game2_tex2D;

texture<int, 2> game_tex2D_array;
texture<int, 2> game1_tex2D_array;
texture<int, 2> game2_tex2D_array;


surface<void, 2> game1_surface;
surface<void, 2> game2_surface;

__host__ __device__ void printgame(int* game, int dim){
    for (int y = 0; y < dim ; y++){
        for (int x = 0 ; x < dim ; x++){
            printf("%d ", game[y*dim + x]);
        }
        printf("\n");
    }
    printf("\n");
}

__host__ __device__ inline int positive_mod(int s, int m) {
    if (s >=0) {
        return s % m;
    } else {
        return m + (s % m);
    }    
}

__host__ __device__ int countneigh(int *game, int x, int y, int dim) {
    int n = 0;
    
    int xp1 = positive_mod(x+1, dim);
    int xm1 = positive_mod(x-1, dim);
    int yp1 = positive_mod(y+1, dim);
    int ym1 = positive_mod(y-1, dim);
    
    n = game[y*dim   + xm1] +
        game[y*dim   + xp1] +
        game[yp1*dim + x] +
        game[ym1*dim + x]+
        game[ym1*dim + xm1] +
        game[yp1*dim + xp1] +
        game[yp1*dim + xm1] +
        game[ym1*dim + xp1] ;
    
    return n;
    
}

void setrandomconfi(int *game, int dim, float p) {
    
    for (int i = 0 ; i < dim*dim ; i++){
        game[i] = ((double) rand() / (RAND_MAX)) < p;
    }
}

void play_game_cpu(int *game_new, int *game_old, int dim) {
    
    // there order, either y first or x first, affects speed of the CPU code quite a bit
    for (int x = 0; x < dim ; x++){
        for (int y = 0 ; y < dim ; y++){
            
            // first copy input to output. Then make transitions.
            game_new[y*dim + x] = game_old[y*dim + x];
            
            int num_neigh_cells = countneigh(game_old, x, y, dim);

            //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
            //Any live cell with more than three live neighbours dies, as if by overpopulation.
            if (game_old[y*dim + x] == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3) ){
                game_new[y*dim + x] = 0;
            }
            //Any live cell with two or three live neighbours lives on to the next generation.
            if (game_old[y*dim + x] ==1 && (num_neigh_cells == 2 || num_neigh_cells == 3) ){
                game_new[y*dim + x] = 1;
            }
            //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
            if (game_old[y*dim + x] == 0 && num_neigh_cells == 3){
                game_new[y*dim + x] = 1;
            }
        }
    }
}

__global__ void play_game_gpu_simple(int *game_new, int *game_old, int dim){
	// Similar to cpu version
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    game_new[y*dim + x] = game_old[y*dim + x];
    int num_neigh_cells = countneigh(game_old, x, y, dim);
    __syncthreads();
    
    //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
    //Any live cell with more than three live neighbours dies, as if by overpopulation.
    if (game_old[y*dim + x] == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3) ){
        game_new[y*dim + x] = 0;
    }
    //Any live cell with two or three live neighbours lives on to the next generation.
    if (game_old[y*dim + x] ==1 && (num_neigh_cells == 2 || num_neigh_cells == 3) ){
        game_new[y*dim + x] = 1;
    }
    //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
    if (game_old[y*dim + x] == 0 && num_neigh_cells == 3){
        game_new[y*dim + x] = 1;
    }
}

__global__ void play_game_gpu_texture1D(int *game_new, int dim, int dir){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (dir) {
        int cell = tex1Dfetch(game1_tex, y * dim + x);
        game_new[y * dim + x] = cell;

        int xp1 = positive_mod(x+1, dim);
        int xm1 = positive_mod(x-1, dim);
        int yp1 = positive_mod(y+1, dim);
        int ym1 = positive_mod(y-1, dim);
        
        int num_neigh_cells = tex1Dfetch(game1_tex, y * dim + xm1) +
        tex1Dfetch(game1_tex, y * dim + xp1) +
        tex1Dfetch(game1_tex, yp1 * dim + x) +
        tex1Dfetch(game1_tex, ym1 * dim + x) +
        tex1Dfetch(game1_tex, ym1 * dim + xm1) +
        tex1Dfetch(game1_tex, yp1 * dim + xp1) +
        tex1Dfetch(game1_tex, yp1 * dim + xm1) +
        tex1Dfetch(game1_tex, ym1 * dim + xp1);
        
        //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        //Any live cell with more than three live neighbours dies, as if by overpopulation.
        if (cell == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3) ){
            game_new[y * dim + x] = 0;
        }
        //Any live cell with two or three live neighbours lives on to the next generation.
        if (cell == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3) ){
            game_new[y * dim + x] = 1;
        }
        //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        if (cell == 0 && num_neigh_cells == 3){
            game_new[y * dim + x] = 1;
        }
    } else {
        int cell = tex1Dfetch(game2_tex, y * dim + x);
        game_new[y * dim + x] = cell;
                
        int xp1 = positive_mod(x+1, dim);
        int xm1 = positive_mod(x-1, dim);
        int yp1 = positive_mod(y+1, dim);
        int ym1 = positive_mod(y-1, dim);
        
        int num_neigh_cells = tex1Dfetch(game2_tex, y * dim + xm1) +
        tex1Dfetch(game2_tex, y * dim + xp1) +
        tex1Dfetch(game2_tex, yp1 * dim + x) +
        tex1Dfetch(game2_tex, ym1 * dim + x) +
        tex1Dfetch(game2_tex, ym1 * dim + xm1) +
        tex1Dfetch(game2_tex, yp1 * dim + xp1) +
        tex1Dfetch(game2_tex, yp1 * dim + xm1) +
        tex1Dfetch(game2_tex, ym1 * dim + xp1);
        
        //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        //Any live cell with more than three live neighbours dies, as if by overpopulation.
        if (cell == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3) ){
            game_new[y*dim + x] = 0;
        }
        //Any live cell with two or three live neighbours lives on to the next generation.
        if (cell == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3) ){
            game_new[y*dim + x] = 1;
        }
        //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        if (cell == 0 && num_neigh_cells == 3){
            game_new[y*dim + x] = 1;
        }
    }
}

__global__ void play_game_gpu_texture2D(int *game_new, int dim, int dir){
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    if (dir) {
        int s = tex2D(game1_tex2D, x, y);
        game_new[y * dim + x] = s;
       	        
        int xp1 = positive_mod(x+1, dim);
        int xm1 = positive_mod(x-1, dim);
        int yp1 = positive_mod(y+1, dim);
        int ym1 = positive_mod(y-1, dim);
        
        int num_neigh_cells = tex2D(game1_tex2D, xm1, y) +
        tex2D(game1_tex2D, xp1, y) +
        tex2D(game1_tex2D, x, yp1) +
        tex2D(game1_tex2D, x, ym1) +
        tex2D(game1_tex2D, xm1, ym1) +
        tex2D(game1_tex2D, xp1, ym1) +
        tex2D(game1_tex2D, xm1, yp1) +
        tex2D(game1_tex2D, xp1, yp1);
        
        //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        //Any live cell with more than three live neighbours dies, as if by overpopulation.
        if (s == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3) ){
            game_new[y*dim + x] = 0;
        }
        //Any live cell with two or three live neighbours lives on to the next generation.
        if (s == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3) ){
            game_new[y*dim + x] = 1;
        }
        //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        if (s == 0 && num_neigh_cells == 3){
            game_new[y*dim + x] = 1;
        }   
    } else {
        int s = tex2D(game2_tex2D, x, y);
        game_new[y*dim + x] = s;
                
        int xp1 = positive_mod(x+1, dim);
        int xm1 = positive_mod(x-1, dim);
        int yp1 = positive_mod(y+1, dim);
        int ym1 = positive_mod(y-1, dim);
        
        int num_neigh_cells = tex2D(game2_tex2D, xm1, y) +
        tex2D(game2_tex2D, xp1, y) +
        tex2D(game2_tex2D, x, yp1) +
        tex2D(game2_tex2D, x, ym1) +
        tex2D(game2_tex2D, xm1, ym1) +
        tex2D(game2_tex2D, xp1, ym1) +
        tex2D(game2_tex2D, xm1, yp1) +
        tex2D(game2_tex2D, xp1, yp1);
        
        //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        //Any live cell with more than three live neighbours dies, as if by overpopulation.
        if (s == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3) ){
            game_new[y*dim + x] = 0;
        }
        //Any live cell with two or three live neighbours lives on to the next generation.
        if (s == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3) ){
            game_new[y*dim + x] = 1;
        }
        //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        if (s == 0 && num_neigh_cells == 3){
            game_new[y*dim + x] = 1;
        }
    }
}

__global__ void play_game_gpu_texture2D_arrays(int *game_new, int dim) {
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    int s = tex2D(game_tex2D_array, x, y);
    game_new[y * dim + x] = s;
            
    int xp1 = positive_mod(x+1, dim);
    int xm1 = positive_mod(x-1, dim);
    int yp1 = positive_mod(y+1, dim);
    int ym1 = positive_mod(y-1, dim);

    int num_neigh_cells = tex2D(game_tex2D_array, xm1, y) +
    tex2D(game_tex2D_array, xp1, y) +
    tex2D(game_tex2D_array, x, yp1) +
    tex2D(game_tex2D_array, x, ym1) +
    tex2D(game_tex2D_array, xm1, ym1) +
    tex2D(game_tex2D_array, xp1, ym1) +
    tex2D(game_tex2D_array, xm1, yp1) +
    tex2D(game_tex2D_array, xp1, yp1);

    //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
    //Any live cell with more than three live neighbours dies, as if by overpopulation.
    if (s == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3) ){
        game_new[y*dim + x] = 0;
    }
    //Any live cell with two or three live neighbours lives on to the next generation.
    if (s == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3) ){
        game_new[y*dim + x] = 1;
    }
    //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
    if (s == 0 && num_neigh_cells == 3){
        game_new[y*dim + x] = 1;
    }
}

__global__ void play_game_gpu_texture2D_surfaces(int dim, int dir){
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    if (dir) {
        int s = tex2D(game1_tex2D_array, x, y);
        surf2Dwrite(s, game2_surface, 4*x, y, cudaBoundaryModeTrap);
                
        int xp1 = positive_mod(x+1, dim);
        int xm1 = positive_mod(x-1, dim);
        int yp1 = positive_mod(y+1, dim);
        int ym1 = positive_mod(y-1, dim);
        
        int num_neigh_cells = tex2D(game1_tex2D_array, xm1, y) +
        tex2D(game1_tex2D_array, xp1, y) +
        tex2D(game1_tex2D_array, x, yp1) +
        tex2D(game1_tex2D_array, x, ym1) +
        tex2D(game1_tex2D_array, xm1, ym1) +
        tex2D(game1_tex2D_array, xp1, ym1) +
        tex2D(game1_tex2D_array, xm1, yp1) +
        tex2D(game1_tex2D_array, xp1, yp1);

        //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        //Any live cell with more than three live neighbours dies, as if by overpopulation.
        if (s == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3) ){
            surf2Dwrite(0, game2_surface, 4*x, y, cudaBoundaryModeTrap);
        }
        //Any live cell with two or three live neighbours lives on to the next generation.
        if (s == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3) ){
            surf2Dwrite(1, game2_surface, 4*x, y, cudaBoundaryModeTrap);
        }
        //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        if (s == 0 && num_neigh_cells == 3){
            surf2Dwrite(1, game2_surface, 4*x, y, cudaBoundaryModeTrap);
        }
    } else {
        int s = tex2D(game2_tex2D_array, x, y);
        surf2Dwrite(s, game1_surface, 4*x, y, cudaBoundaryModeTrap);
                
        int xp1 = positive_mod(x+1, dim);
        int xm1 = positive_mod(x-1, dim);
        int yp1 = positive_mod(y+1, dim);
        int ym1 = positive_mod(y-1, dim);
        
        int num_neigh_cells = tex2D(game2_tex2D, xm1, y) +
        tex2D(game2_tex2D_array, xp1, y) +
        tex2D(game2_tex2D_array, x, yp1) +
        tex2D(game2_tex2D_array, x, ym1) +
        tex2D(game2_tex2D_array, xm1, ym1) +
        tex2D(game2_tex2D_array, xp1, ym1) +
        tex2D(game2_tex2D_array, xm1, yp1) +
        tex2D(game2_tex2D_array, xp1, yp1);
        
        //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        //Any live cell with more than three live neighbours dies, as if by overpopulation.
        if (s == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3) ){
            surf2Dwrite(0, game1_surface, 4*x, y, cudaBoundaryModeTrap);
        }
        //Any live cell with two or three live neighbours lives on to the next generation.
        if (s == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3) ){
            surf2Dwrite(1, game1_surface, 4*x, y, cudaBoundaryModeTrap);
        }
        //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        if (s == 0 && num_neigh_cells == 3){
            surf2Dwrite(1, game1_surface, 4*x, y, cudaBoundaryModeTrap);
        }
    }
}

int main() {
	// Set environment
    float cpu_time, gpu_time;
    int error = 0;
    int num_iterations = 100;
    int *original_game = (int *) malloc(sizeof(int) * GAMESIZE);
    setrandomconfi(original_game, GAMEDIM, 0.6);
    //printgame(original_game,GAMEDIM);

    /* 
    	CPU version
    */
    int *host_game_1 = (int *) malloc(sizeof(int) * GAMESIZE);
    int *host_game_2 = (int *) malloc(sizeof(int) * GAMESIZE);
    for (int i = 0; i < GAMESIZE; i++) {
    	host_game_1[i] = original_game[i];
    }
    cstart();
    for (int t = 1; t <= num_iterations/2 ; t++){
        play_game_cpu(host_game_2, host_game_1, GAMEDIM);
        play_game_cpu(host_game_1, host_game_2, GAMEDIM);
    }
    cend(&cpu_time);
    printf("cpu time = %f\n", cpu_time);

    /* 
    	GPU versions
    */

    // Set enviroment
    dim3 threads_per_block(32, 32);
    dim3 num_blocks((GAMEDIM + 31) / 32,(GAMEDIM+31) / 32);

    int *dev_game1, *dev_game2;
    cudaMalloc(&dev_game1, sizeof(int)*GAMESIZE);
    cudaMalloc(&dev_game2, sizeof(int)*GAMESIZE);
    cudaMemcpy(dev_game1, original_game, sizeof(int)*GAMESIZE, cudaMemcpyHostToDevice);

    gstart();
    for (int t = 1; t <= num_iterations/2 ; t++){
        play_game_gpu_simple<<<num_blocks, threads_per_block>>>(dev_game2, dev_game1, GAMEDIM);
        play_game_gpu_simple<<<num_blocks, threads_per_block>>>(dev_game1, dev_game2, GAMEDIM);
    }
    gend(&gpu_time);
    cudaMemcpy(host_game_2, dev_game1, sizeof(int)*GAMESIZE, cudaMemcpyDeviceToHost);
    error = 0;
    for (int i = 0; i < GAMESIZE; i++) {
    	error += (host_game_2[i] - host_game_1[i] + 2) % 2;
    }
    printf("gpu time (simple) = %f, error is %d\n", gpu_time, error);

    // Use 1D texture memory
    cudaMemcpy(dev_game1, original_game, sizeof(int)*GAMESIZE, cudaMemcpyHostToDevice);
    cudaBindTexture(NULL, game1_tex, dev_game1, GAMESIZE * sizeof(int));
    cudaBindTexture(NULL, game2_tex, dev_game2, GAMESIZE * sizeof(int));
    gstart();
    for (int t = 1; t <= num_iterations/2 ; t++){
        play_game_gpu_texture1D<<<num_blocks, threads_per_block>>>(dev_game2, GAMEDIM, 1);
        play_game_gpu_texture1D<<<num_blocks, threads_per_block>>>(dev_game1, GAMEDIM, 0);
    }
    gend(&gpu_time);
    cudaMemcpy(host_game_2, dev_game1, GAMESIZE * sizeof(int), cudaMemcpyDeviceToHost);
    error = 0;
    for (int i = 0; i < GAMESIZE; i++) {
    	error += (host_game_2[i] - host_game_1[i] + 2) % 2;
    }
    printf("gpu time (texture1D) = %f, error is %d\n", gpu_time, error);

    // Use 2D texture memory
    cudaMemcpy(dev_game1, original_game, sizeof(int)*GAMESIZE, cudaMemcpyHostToDevice);
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    cudaBindTexture2D(NULL, game1_tex2D, dev_game1, desc, GAMEDIM, GAMEDIM, sizeof(int)*GAMEDIM);
    cudaBindTexture2D(NULL, game2_tex2D, dev_game2, desc, GAMEDIM, GAMEDIM, sizeof(int)*GAMEDIM);
    gstart();
    for (int t = 1; t <= num_iterations/2 ; t++){
        play_game_gpu_texture2D<<<num_blocks, threads_per_block>>>(dev_game2, GAMEDIM, 1);
        play_game_gpu_texture2D<<<num_blocks, threads_per_block>>>(dev_game1, GAMEDIM, 0);
    }
    gend(&gpu_time);
    cudaMemcpy(host_game_2, dev_game1, GAMESIZE * sizeof(int), cudaMemcpyDeviceToHost);
    error = 0;
    for (int i = 0; i < GAMESIZE; i++) {
    	error += (host_game_2[i] - host_game_1[i] + 2) % 2;
    }
    printf("gpu time (texture2D) = %f, error is %d\n", gpu_time, error);

    // Use 2D texture array
    cudaArray *dev_arr;
    cudaMallocArray(&dev_arr, &desc, GAMEDIM, GAMEDIM, cudaArraySurfaceLoadStore);
    cudaBindTextureToArray(game_tex2D_array, dev_arr, desc);
    cudaMemcpyToArray(dev_arr, 0, 0, original_game, sizeof(int)*GAMESIZE, cudaMemcpyHostToDevice);
    gstart();
    for (int t = 1; t <= num_iterations ; t++){
        play_game_gpu_texture2D_arrays<<<num_blocks, threads_per_block>>>(dev_game1, GAMEDIM);
        cudaMemcpyToArray(dev_arr, 0,0, dev_game1, GAMESIZE * sizeof(int), cudaMemcpyDeviceToDevice);
    }
    gend(&gpu_time);
    cudaMemcpy(host_game_2, dev_game1, GAMESIZE * sizeof(int), cudaMemcpyDeviceToHost);
    error = 0;
    for (int i = 0; i < GAMESIZE; i++) {
        error += (host_game_2[i] - host_game_1[i] + 2) % 2;
    }
    printf("gpu time (texture2D array) = %f, error is %d\n", gpu_time, error);

    // Use 2D texture surfaces
    cudaArray *dev_arr1;
    cudaArray *dev_arr2;
    cudaMallocArray(&dev_arr1, &desc, GAMEDIM, GAMEDIM, cudaArraySurfaceLoadStore);
    cudaMallocArray(&dev_arr2, &desc, GAMEDIM, GAMEDIM, cudaArraySurfaceLoadStore);
    cudaBindTextureToArray(game1_tex2D_array, dev_arr1, desc);
    cudaBindTextureToArray(game2_tex2D_array, dev_arr2, desc);
    cudaBindSurfaceToArray(game1_surface, dev_arr1);
    cudaBindSurfaceToArray(game2_surface, dev_arr2);
    cudaMemcpyToArray(dev_arr1, 0, 0, original_game, sizeof(int)*GAMESIZE, cudaMemcpyHostToDevice);
    gstart();
    for (int t = 1; t <= num_iterations/2 ; t++){
        play_game_gpu_texture2D_surfaces<<<num_blocks, threads_per_block>>>(GAMEDIM, 1);
        play_game_gpu_texture2D_surfaces<<<num_blocks, threads_per_block>>>(GAMEDIM, 0);
    }
    gend(&gpu_time);
    cudaMemcpyFromArray(host_game_2, dev_arr1, 0,0, GAMESIZE * sizeof(int), cudaMemcpyDeviceToHost);
    error = 0;
    for (int i = 0; i < GAMESIZE; i++) {
        error += (host_game_2[i] - host_game_1[i] + 2) % 2;
    }
    printf("gpu time (texture2D surfaces) = %f, error is %d\n", gpu_time, error);

    free(original_game);
    free(host_game_1);
    free(host_game_2);
    cudaFree(dev_game1);
    cudaFree(dev_game2);

	return 0;
}
