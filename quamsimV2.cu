#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;
__global__ void quantumGate(const float *U, const float *A, float *B, const int a_size, const int *qubit, const int *non_qubit) {

	__shared__ float shared_array[64];		// shared array in the thread block. Size is always 64 (2^6)
	int block_offset;					// The unchanging (in a block) part of index, given by block ID
	block_offset = 0;

	int thread_id = threadIdx.x;
	int block_id = blockIdx.x;
	int qubit_index;											// The changing part of 64 index, given by qubits.

	for(int j = 0; j<log2((double)((a_size))-6); j++){
		block_offset += ((block_id >> j) & 1) << non_qubit[j];	// get the block offset for shared array by shifting the block id bits to it's corresponding position
	}																													// this number is universal for the same thread block

	qubit_index = (((thread_id >> 0) & 1) << qubit[0]) +
								(((thread_id >> 1) & 1) << qubit[1]) +
								(((thread_id >> 2) & 1) << qubit[2]) +
								(((thread_id >> 3) & 1) << qubit[3]) +
								(((thread_id >> 4) & 1) << qubit[4]) +
								(((thread_id >> 5) & 1) << qubit[5]);
	shared_array[thread_id] = A[qubit_index + block_offset];	
	qubit_index = ((((thread_id+32) >> 0) & 1) << qubit[0]) +
								((((thread_id+32) >> 1) & 1) << qubit[1]) +
								((((thread_id+32) >> 2) & 1) << qubit[2]) +
								((((thread_id+32) >> 3) & 1) << qubit[3]) +
								((((thread_id+32) >> 4) & 1) << qubit[4]) +
								((((thread_id+32) >> 5) & 1) << qubit[5]);
	shared_array[thread_id+32] = A[qubit_index + block_offset];	
	//if (thread_id%32 == 0) printf("initial shared_array[%d]: %f\n", j, shared_array[j]);
	
	__syncthreads();

	int apply_id_0;
	int apply_id_1;
	for (int qubit_applied = 0; qubit_applied<6; qubit_applied++){
		apply_id_0 = thread_id*2 - thread_id%(1<<qubit_applied);						// algriothm for matching shared_array index to be used and thread id.
		apply_id_1 = thread_id*2 - thread_id%(1<<qubit_applied) + (1<<qubit_applied); 

		float temp_0 = shared_array[apply_id_0];
		float temp_1 = shared_array[apply_id_1];

		shared_array[apply_id_0] = U[4*qubit_applied+0] * temp_0 + U[4*qubit_applied+1] * temp_1;		// matrix multiplication
		shared_array[apply_id_1] = U[4*qubit_applied+2] * temp_0 + U[4*qubit_applied+3] * temp_1;
		//printf("result shared_array[%d]: %f\n", apply_id_0, shared_array[apply_id_0]);
		//printf("result shared_array[%d]: %f\n", apply_id_1, shared_array[apply_id_1]);
		__syncthreads();
	}

	int wb_id_0 = (((apply_id_0 >> 0) & 1) << qubit[0]) +														// convert the index of shared_array to result array index.
								(((apply_id_0 >> 1) & 1) << qubit[1]) +
								(((apply_id_0 >> 2) & 1) << qubit[2]) +
								(((apply_id_0 >> 3) & 1) << qubit[3]) +
								(((apply_id_0 >> 4) & 1) << qubit[4]) +
								(((apply_id_0 >> 5) & 1) << qubit[5]) +
								block_offset;

	int wb_id_1 = (((apply_id_1 >> 0) & 1) << qubit[0]) +
								(((apply_id_1 >> 1) & 1) << qubit[1]) +
								(((apply_id_1 >> 2) & 1) << qubit[2]) +
								(((apply_id_1 >> 3) & 1) << qubit[3]) +
								(((apply_id_1 >> 4) & 1) << qubit[4]) +
								(((apply_id_1 >> 5) & 1) << qubit[5]) +
								block_offset;	

	B[wb_id_0] = shared_array[apply_id_0];
	B[wb_id_1] = shared_array[apply_id_1];
	//printf("output array: B[%d] = %f\n", wb_id_0, B[wb_id_0]);
	//printf("output array: B[%d] = %f\n", wb_id_1, B[wb_id_1]);
}

bool contain(int *qubit, int index){
	for(int i = 0; i < 6; i++){
		if(qubit[i] == index) return true;
	}
	return false;
}

int main (int argc, char **argv) {
	// read arguments, open input file
	ifstream input_file;
	if (argc != 2) {
		cerr << "# of arguments error, please enter input file name" << endl;
		exit(EXIT_FAILURE);
	}
	else {
		input_file.open(argv[1]);
	}
	//cout << "filename: " << argv[1] << endl;
	// parse input file, store data
	int n_qubit = 6;
	float *h_matrix_u = (float *)malloc(sizeof(float)*4*n_qubit);
	float *h_array_a = (float *)malloc(sizeof(float)*pow(2, 30));
	int *h_qubit = (int *)malloc(sizeof(int)*n_qubit);
	int parse_flag = 0;
	int i = 0;
	int j = 0;
	string line;

	while(getline(input_file, line)){
		if (line.empty()) parse_flag++ ;
		else if (parse_flag < 12 && parse_flag%2 == 0){
			h_matrix_u[parse_flag*2] = stof(line.substr(0, line.find(" ")));
			h_matrix_u[parse_flag*2+1] = stof(line.substr(line.find(" ")+1, line.find("\n")));
			parse_flag ++;
		}
		else if (parse_flag < 12 && parse_flag%2 == 1){
			h_matrix_u[parse_flag*2] = stof(line.substr(0, line.find(" ")));
			h_matrix_u[parse_flag*2+1] = stof(line.substr(line.find(" ")+1, line.find("\n")));
		}
		else if (parse_flag == 12){
			h_array_a[i] = stof(line);
			i++;
		}
		else if (parse_flag == 13){
			h_qubit[j] = stoi(line);
			j++;
		}
	}
	int a_size = i;
	//cout << "array size:	" << a_size << endl;
	int n_non_qubit = log2(a_size) - n_qubit;
	//cout << "non_qubit #:	" << n_non_qubit << endl;

	h_array_a = (float *)realloc(h_array_a, a_size*sizeof(float));
	float *h_array_b = (float *)malloc(a_size*sizeof(float));

	int *h_non_qubit = (int *)malloc(sizeof(int)*n_non_qubit);

	for(i=0; i<n_non_qubit; ){
		for(j=0; j<log2(a_size); j++){
			if (!contain(h_qubit, j) && i<n_non_qubit) {
				h_non_qubit[i] = j;
				i++;
			}
		}
	}

	//cout << "array size: " << a_size/2 << endl;
	//cout << "qubit: " << qubit[0]  << qubit[1]  << qubit[2] << endl;
	//cuda related part
  //cudaEvent_t start, stop;
  //cudaEventCreate (&start);
  //cudaEventCreate (&stop);
  cudaError_t err = cudaSuccess;
  // Allocate the device input matrix U
	float *d_matrix_u = NULL;
	err = cudaMalloc((void **)&d_matrix_u, n_qubit*4*sizeof(float));
  if (err != cudaSuccess) {
    cerr << "Failed to allocate device matrix U (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }

  // Allocate the device input array A
	float *d_array_a = NULL;
	err = cudaMalloc((void **)&d_array_a, a_size*sizeof(float));
  if (err != cudaSuccess) {
    cerr << "Failed to allocate device array A (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }

	// Allocate the device output array B
	float *d_array_b = NULL;
	err = cudaMalloc((void **)&d_array_b, a_size*sizeof(float));
  if (err != cudaSuccess) {
    cerr << "Failed to allocate device array B (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }

	int *d_qubit = NULL;
	err = cudaMalloc((void **)&d_qubit, n_qubit*sizeof(int));
  if (err != cudaSuccess) {
    cerr << "Failed to allocate device qubit (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }

	int *d_non_qubit = NULL;
	err = cudaMalloc((void **)&d_non_qubit, n_non_qubit*sizeof(int));
  if (err != cudaSuccess) {
    cerr << "Failed to allocate device non qubit (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }
  // Copy the host input matrix U in host memory to the device input vectors in device memory
	err = cudaMemcpy(d_matrix_u, h_matrix_u, n_qubit*4*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cerr << "Failed to copy matrix U from host to device (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }

	// Copy the host input array A in host memory to the device input vectors in device memory
	err = cudaMemcpy(d_array_a, h_array_a, a_size*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
		cerr << "Failed to copy array A from host to device (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }

	err = cudaMemcpy(d_qubit, h_qubit, n_qubit*sizeof(int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
		cerr << "Failed to copy qubit from host to device (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }

	err = cudaMemcpy(d_non_qubit, h_non_qubit, n_non_qubit*sizeof(int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
		cerr << "Failed to copy non-qubit from host to device (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }

	int threadsPerBlock = 32;
  int blocksPerGrid =(a_size/2 + threadsPerBlock - 1) / threadsPerBlock;
  //cudaEventRecord(start);

	quantumGate<<<blocksPerGrid, threadsPerBlock>>>(d_matrix_u, d_array_a, d_array_b, a_size, d_qubit, d_non_qubit);

  //cudaEventRecord(stop);
	err = cudaGetLastError();
  if (err != cudaSuccess) {
	cerr << "Failed to launch quantumGate kernel (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(h_array_b, d_array_b, a_size*sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    cerr << "Failed to copy array B from device to host (error code " << cudaGetErrorString(err) << ")!" << endl;
	}
	
	//cudaEventSynchronize(stop);
	//float run_time = 0;
	//cudaEventElapsedTime(&run_time, start, stop);
	// cout << "Malloc & Memcpy run time: " << run_time << endl;
	// print result to screen
	for(i=0; i<a_size; i++) cout << fixed << setprecision(3) << h_array_b[i] << endl;

    // Free host memory
	free(h_matrix_u);
	free(h_array_a);
	free(h_array_b);
	free(h_qubit);
	free(h_non_qubit);

  // Free device global memory
  err = cudaFree(d_matrix_u);
  if (err != cudaSuccess) {
    cerr << "Failed to free device matrix U (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }
  err = cudaFree(d_array_a);
  if (err != cudaSuccess) {
    cerr << "Failed to free device array A (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }
  err = cudaFree(d_array_b);
  if (err != cudaSuccess) {
    cerr << "Failed to free device array B (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_qubit);
  if (err != cudaSuccess) {
    cerr << "Failed to free device qubit (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }
  err = cudaFree(d_non_qubit);
  if (err != cudaSuccess) {
    cerr << "Failed to free device non-qubit (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }

	err = cudaDeviceReset();
  if (err != cudaSuccess) {
    cerr << "Failed to deinitialize the device! error=" << cudaGetErrorString(err) << endl;
    exit(EXIT_FAILURE);
  }

	return 0;
}
