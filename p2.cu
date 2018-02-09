#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define N 12

// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void cuadrados_GPU(int *dev_cuadrados) {
	int id = threadIdx.x;
	dev_cuadrados[id] = id*2+1;
}

// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv) {
	// declaraciones
	int *hst_salida;
	int *dev_salida;
	// reserva en el host
	hst_salida = (int*)malloc(N*sizeof(int));
	// reserva en el device
	cudaMalloc((void**)&dev_salida, N*sizeof(int));
	// llamada a la funcion cuadrados_GPU
	cuadrados_GPU <<<1,N>>>(dev_salida);
	// recogida de datos desde el device
	cudaMemcpy(hst_salida, dev_salida, N*sizeof(int), cudaMemcpyDeviceToHost);
	// resultados GPU
	printf("LOS %d PRIMEROS NUMEROS IMPARES SON:\n", N);
	for (int i = 0; i < N; i++) printf("[%2d] -> %d\n", i, hst_salida[i]);
	// salida
	printf("\npulsa INTRO para finalizar...");
  fflush(stdin);
	char tecla = getchar();
	return 0;
}
