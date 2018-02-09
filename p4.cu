#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOQUE 5

__global__ void info (int *vector, int *resultado, int desplazamiento, int tamano) {
	int gId = threadIdx.x + blockDim.x * blockIdx.x;
	if (gId<tamano)
		resultado[gId] = vector[((-desplazamiento/tamano + 1)*tamano+gId+desplazamiento)%tamano];
}

int main(int argc, char** argv) {
	printf("Introduce el tamaño del vector: ");
	int tamano;
	scanf("%d",&tamano);
	printf("\nIntroduce el desplazamiento: ");
	int desplazamiento;
	scanf("%d",&desplazamiento);
	printf("\n\n tamano: %d\n", tamano);

	int *vector, *resultado;
	int *dev_vector, *dev_resultado; // reserva en el host
	
	// declaracion de eventos
	cudaEvent_t start;
	cudaEvent_t stop;
	// creacion de eventos
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	vector = (int *)malloc(tamano*sizeof(int));
	resultado = (int *)malloc(tamano*sizeof(int));

	// reserva en el device
	cudaMalloc( (void**)&dev_vector, tamano*sizeof(int));
	cudaMalloc( (void**)&dev_resultado, tamano*sizeof(int));
	for(int i=0;i<tamano;i++){
		vector[i]=2*i+1;
	}
	cudaMemcpy(dev_vector, vector, tamano*sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(start,0);
	info<<<(tamano + BLOQUE - 1)/BLOQUE,BLOQUE>>>(dev_vector, dev_resultado, desplazamiento, tamano);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime,start,stop);

	cudaMemcpy(resultado, dev_resultado, tamano*sizeof(int), cudaMemcpyDeviceToHost);

	printf("Vector de entrada\n");
	for(int i=0;i<tamano;i++){
		printf("%2d ", vector[i]);
	}
  
	printf("\nVector de salida\n");
	for(int i=0;i<tamano;i++){
		printf("%2d ", resultado[i]);
	}

	printf("\n> Tiempo de ejecucion: %f ms\n",elapsedTime);

	// liberacion de recursos
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree( dev_resultado );
	cudaFree( dev_vector );

	// salida
	printf("\npulsa INTRO para finalizar..."); fflush(stdin);
	char tecla = getchar();
	return 0;
}
