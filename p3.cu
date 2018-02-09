#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define HILOS 24
#define BLOQUES 4

// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)+
__global__ void info (int *resultado1, int *resultado2, int *resultado3) {
	int gId = threadIdx.x + blockDim.x * blockIdx.x;
	int id = threadIdx.x;
	int bId = blockIdx.x;
	resultado1[gId] = id;
	resultado2[gId] = bId;
	resultado3[gId] = gId;

}

// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv) {
	// declaraciones
	int *resultado1, *resultado2, *resultado3;
	int *dev_resultado1, *dev_resultado2, *dev_resultado3; // reserva en el host
	resultado1 = (int *)malloc(HILOS*sizeof(int));
	resultado2 = (int *)malloc(HILOS*sizeof(int));
	resultado3 = (int *)malloc(HILOS*sizeof(int));

	// reserva en el device
	cudaMalloc( (void**)&dev_resultado1, HILOS*sizeof(int));
	cudaMalloc( (void**)&dev_resultado2, HILOS*sizeof(int));
	cudaMalloc( (void**)&dev_resultado3, HILOS*sizeof(int));

	info<<<1,HILOS>>>(dev_resultado1, dev_resultado2, dev_resultado3);
	// recogida de datos desde el device
	cudaMemcpy(resultado1, dev_resultado1, HILOS*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(resultado2, dev_resultado2, HILOS*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(resultado3, dev_resultado3, HILOS*sizeof(int), cudaMemcpyDeviceToHost);

	// impresion de resultados
	printf("\n\nLanzamiento con %2d bloques de %2d hilos", 1, HILOS);
	printf( "\nIndice de hilo:\n");
	for (int i = 0; i < HILOS; i++) printf("%2d ", resultado1[i]);
	printf( "\nIndice de bloque:\n");
	for (int i = 0; i < HILOS; i++) printf("%2d ", resultado2[i]);
	printf( "\nIndice global:\n");
	for (int i = 0; i < HILOS; i++) printf("%2d ", resultado3[i]);
	
	info<<<HILOS,1>>>(dev_resultado1, dev_resultado2, dev_resultado3);
	// recogida de datos desde el device
	cudaMemcpy(resultado1, dev_resultado1, HILOS*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(resultado2, dev_resultado2, HILOS*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(resultado3, dev_resultado3, HILOS*sizeof(int), cudaMemcpyDeviceToHost);

	// impresion de resultados
	printf("\n\nLanzamiento con %2d bloques de %2d hilos", HILOS, 1);
	printf( "\nIndice de hilo:\n");
	for (int i = 0; i < HILOS; i++) printf("%2d ", resultado1[i]);
	printf( "\nIndice de bloque:\n");
	for (int i = 0; i < HILOS; i++) printf("%2d ", resultado2[i]);
	printf( "\nIndice global:\n");
	for (int i = 0; i < HILOS; i++) printf("%2d ", resultado3[i]);

	info<<<BLOQUES, HILOS/BLOQUES>>>(dev_resultado1, dev_resultado2, dev_resultado3);
	// recogida de datos desde el device
	cudaMemcpy(resultado1, dev_resultado1, HILOS*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(resultado2, dev_resultado2, HILOS*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(resultado3, dev_resultado3, HILOS*sizeof(int), cudaMemcpyDeviceToHost);

	// impresion de resultados
	printf("\n\nLanzamiento con %2d bloques de %2d hilos", BLOQUES, HILOS/BLOQUES);
	printf( "\nIndice de hilo:\n");
	for (int i = 0; i < HILOS; i++) printf("%2d ", resultado1[i]);
	printf( "\nIndice de bloque:\n");
	for (int i = 0; i < HILOS; i++) printf("%2d ", resultado2[i]);
	printf( "\nIndice global:\n");
	for (int i = 0; i < HILOS; i++) printf("%2d ", resultado3[i]);

	// liberamos memoria en el device
	cudaFree( dev_resultado1 );
	cudaFree( dev_resultado2 );
	cudaFree( dev_resultado3 );

	// salida
	printf("\n\npulsa INTRO para finalizar...");
	fflush(stdin);
	char tecla = getchar(); return 0;
}
