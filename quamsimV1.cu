#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;
__global__ void quantumGate(const float *U, const float *A, float *B, const int a_size, const int qubit) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < a_size) {
		int a1 = (i >> qubit) << (qubit+1);
		int a2 = i & ((1<<qubit) - 1);
		int b0 = (a1|a2) | (0<<qubit);
		int b1 = (a1|a2) | (1<<qubit);
		B[b0] = U[0]*A[b0] + U[1]*A[b1];
		B[b1] = U[2]*A[b0] + U[3]*A[b1];
		//printf("i: %d, j: %d, b0: %d, b1: %d, A[0]: %f, A[1]: %f B[0]: %f, B[1]: %f\n", i, j, b0, b1, A[b0], A[b1], B[b0], B[b1]);
  }
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
	float *h_matrix_u = (float *)malloc(sizeof(float)*4*6);
	float *h_array_a = (float *)malloc(sizeof(float)*pow(2, 30));
	int *qubit = (int *)malloc(sizeof(int)*6);

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
			qubit[j] = stoi(line);
			j++;
		}
	}

	int a_size = i;
	h_array_a = (float *)realloc(h_array_a, a_size*sizeof(float));
	float *h_array_b = (float *)malloc(a_size*sizeof(float));

	//cout << "array size: " << a_size/2 << endl;
	//cout << "qubit: " << qubit[0]  << qubit[1]  << qubit[2] << endl;

	//cuda related part
  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);
  cudaError_t err = cudaSuccess;

  // Allocate the device input matrix U
	float *d_matrix_u = NULL;
	err = cudaMalloc((void **)&d_matrix_u, 6*4*sizeof(float));
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
	float *d_array_b_1_2 = NULL;
	err = cudaMalloc((void **)&d_array_b_1_2, a_size*sizeof(float));
  if (err != cudaSuccess) {
    cerr << "Failed to allocate device array B (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }
	float *d_array_b_2_3 = NULL;
	err = cudaMalloc((void **)&d_array_b_2_3, a_size*sizeof(float));
  if (err != cudaSuccess) {
    cerr << "Failed to allocate device array B (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }
	float *d_array_b_3_4 = NULL;
	err = cudaMalloc((void **)&d_array_b_3_4, a_size*sizeof(float));
  if (err != cudaSuccess) {
    cerr << "Failed to allocate device array B (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }
	float *d_array_b_4_5 = NULL;
	err = cudaMalloc((void **)&d_array_b_4_5, a_size*sizeof(float));
  if (err != cudaSuccess) {
    cerr << "Failed to allocate device array B (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }
	float *d_array_b_5_6 = NULL;
	err = cudaMalloc((void **)&d_array_b_5_6, a_size*sizeof(float));
  if (err != cudaSuccess) {
    cerr << "Failed to allocate device array B (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }
	

  // Copy the host input matrix U in host memory to the device input vectors in device memory
	err = cudaMemcpy(d_matrix_u, h_matrix_u, 6*4*sizeof(float), cudaMemcpyHostToDevice);
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

	int threadsPerBlock = 256;
  int blocksPerGrid =(a_size/2 + threadsPerBlock - 1) / threadsPerBlock;
  cudaEventRecord(start);

	quantumGate<<<blocksPerGrid, threadsPerBlock>>>(d_matrix_u, d_array_a, d_array_b_1_2, a_size/2, qubit[0]);
	cudaDeviceSynchronize();
	quantumGate<<<blocksPerGrid, threadsPerBlock>>>(d_matrix_u+4, d_array_b_1_2, d_array_b_2_3, a_size/2, qubit[1]);
	cudaDeviceSynchronize();
	quantumGate<<<blocksPerGrid, threadsPerBlock>>>(d_matrix_u+8, d_array_b_2_3, d_array_b_3_4, a_size/2, qubit[2]);
	cudaDeviceSynchronize();
	quantumGate<<<blocksPerGrid, threadsPerBlock>>>(d_matrix_u+12, d_array_b_3_4, d_array_b_4_5, a_size/2, qubit[3]);
	cudaDeviceSynchronize();
	quantumGate<<<blocksPerGrid, threadsPerBlock>>>(d_matrix_u+16, d_array_b_4_5, d_array_b_5_6, a_size/2, qubit[4]);
	cudaDeviceSynchronize();
	quantumGate<<<blocksPerGrid, threadsPerBlock>>>(d_matrix_u+20, d_array_b_5_6, d_array_b, a_size/2, qubit[5]);
	cudaDeviceSynchronize();

  cudaEventRecord(stop);
	err = cudaGetLastError();
  if (err != cudaSuccess) {
	cerr << "Failed to launch quantumGate kernel (error code " << cudaGetErrorString(err) << ")!" << endl;
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(h_array_b, d_array_b, a_size*sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    cerr << "Failed to copy array B from device to host (error code " << cudaGetErrorString(err) << ")!" << endl;
	}
	
	cudaEventSynchronize(stop);
	float run_time = 0;
	cudaEventElapsedTime(&run_time, start, stop);
	// cout << "Malloc & Memcpy run time: " << run_time << endl;
	// print result to screen
	for(i=0; i<a_size; i++) cout << fixed << setprecision(3) << h_array_b[i] << endl;

    // Free host memory
	free(h_matrix_u);
	free(h_array_a);
	free(h_array_b);
	free(qubit);

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

	err = cudaDeviceReset();
  if (err != cudaSuccess) {
    cerr << "Failed to deinitialize the device! error=" << cudaGetErrorString(err) << endl;
    exit(EXIT_FAILURE);
  }

	return 0;
}
