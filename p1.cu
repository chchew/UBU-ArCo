// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#define N 8
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv) {
	// declaracion
	float *hst_A;
	float *hst_B;
	float *dev_A;
	float *dev_B;
	// reserva en el host
	hst_A = (float*)malloc( N*sizeof(float) );
	hst_B = (float*)malloc( N*sizeof(float) );
	// reserva en el device
	cudaMalloc( (void**)&dev_A, N*sizeof(float) );
	cudaMalloc( (void**)&dev_B, N*sizeof(float) );
	// inicializacion de datos
	srand ( (int)time(NULL) );
	for (int i=0; i<N; i++) {
		hst_A[i] = (float) rand() / RAND_MAX;
	}
	// copia de datos
	cudaMemcpy(dev_A, hst_A, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, dev_A, N*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaFree( dev_A );
	cudaMemcpy(hst_B, dev_B, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree( dev_B );
	// salida
	int i;
	printf("ENTRADA:\n\n");
	for (i = 0; i < N; i++) printf("%d: %.2f\n", i+1, hst_A[i]);
	printf("SALIDA:\n\n");
	for (i = 0; i < N; i++) printf("%d: %.2f\n", i+1, hst_B[i]);
	printf("\npulsa INTRO para finalizar..."); fflush(stdin);
	char tecla = getchar();
	return 0;
}
