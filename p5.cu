#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>
#include <cuda_runtime.h>

#define MEGA (1<<20)
#define KILO (1<<10)

void dispositivos();
int clean_stdin();
int siguiente_potencia2(int);

__global__ void elementos(float *operacion, int tamano, int precision) {
	int gId = threadIdx.x + blockDim.x * blockIdx.x;
	extern __shared__ float resultado[];
	if (gId < precision)
		resultado[gId] = (float) 6/((gId+1)*(gId+1));
	else
		resultado[gId] = 0;
	__syncthreads();
	int salto = tamano/2;
	while(salto != 0) {
		if(gId < salto){
			resultado[gId] = resultado[gId] + resultado[gId+salto];
		}
		__syncthreads();
		salto = salto/2;
	}
	operacion[gId] = sqrt(resultado[gId]);
}

int main(int argc, char** argv) {
	SetConsoleTitle("ARCO - PRACTICA 5");
	dispositivos();
	int precision = 0;
	while (precision <= 0) {
		printf("Introduce la precision: ");
		char enter;
		int res = scanf("%d%c", &precision, &enter);
		if (res != 2 || enter != '\n'){
			precision = 0;
			clean_stdin();
		}
	}
	int tamano = siguiente_potencia2(precision);
	printf("\n\n> Numero de hilos lanzados: %d\n", tamano);


	float *resultado;
	float *dev_resultado; // reserva en el host
	
	// declaracion de eventos
	cudaEvent_t start;
	cudaEvent_t stop;
	// creacion de eventos
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	resultado = (float *)malloc(tamano*sizeof(float));	
	
	// reserva en el device

	cudaMalloc( (void**)&dev_resultado, tamano*sizeof(float));

	cudaMemcpy(dev_resultado, resultado, tamano*sizeof(float), cudaMemcpyHostToDevice);

	cudaEventRecord(start,0);
	elementos<<<1,tamano,tamano*sizeof(float)>>>(dev_resultado, tamano, precision);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime,start,stop);

	cudaMemcpy(resultado, dev_resultado, tamano*sizeof(float), cudaMemcpyDeviceToHost);

	printf("\nPi = %f", resultado[0]);

	printf("\n\n> Tiempo de ejecucion: %f ms\n",elapsedTime);

	// liberacion de recursos
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree( dev_resultado );

	// salida
	printf("\npulsa INTRO para finalizar..."); fflush(stdin);
	char tecla = getchar();
	return 0;
}

void dispositivos() {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
	printf("!!!!!No se han encontrado dispositivos CUDA!!!!!\n");
	return;
	} else {
	printf("***************************************************\n");
	printf("Se han encontrado <%d> dispositivos CUDA:\n", deviceCount); }
	// declaraciones
	cudaDeviceProp deviceProp;
	size_t SelectedMemory = 0;
	int SelectedDevice = 0;
	for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
		cudaGetDeviceProperties(&deviceProp, currentDevice);
		// calculo del numero de cores
		int cudaCores = 0;
		int SM = deviceProp.multiProcessorCount;
		int major = deviceProp.major;
		int minor = deviceProp.minor;
		switch (major){
			case 1:
			// TESLA
			cudaCores = 8;
			break;
			case 2:
			// FERMI
			if (minor == 0)
			cudaCores = 32;
			else
			cudaCores = 48;
			break;
			case 3:
			//KEPLER
			cudaCores = 192;
			break;
			case 5:
			//MAXWELL
			cudaCores = 128;
			break;
			case 6:
			//PASCAL
				cudaCores = 64;
			break;
			default:
			//ARQUITECTURA DESCONOCIDA
			cudaCores = 0;
			printf("!!!!!dispositivo desconocido!!!!!\n");
		}
		// seleccion del dispositivo con mayor memoria global
		size_t ActualMemory = deviceProp.totalGlobalMem;
		if (ActualMemory>SelectedMemory) {
			SelectedMemory = ActualMemory;
			SelectedDevice = currentDevice;
		}
		// presentacion de propiedades
		printf("***************************************************\n");
		printf("DEVICE %d: %s\n", currentDevice, deviceProp.name);
		printf("***************************************************\n");
		printf("> Compute Capability \t: %d.%d\n", major, minor);
		printf("> No. of Multi Processors \t: %d \n", SM);
		printf("> No. of CUDA Cores (%dx%d)\t: %d \n", cudaCores, SM, cudaCores*SM);
		printf("> Global Memory (total) \t: %u MiB\n", deviceProp.totalGlobalMem/MEGA);
		printf("> Shared Memory (per block)\t: %u KiB\n", deviceProp.sharedMemPerBlock/KILO); printf("> Constant Memory (total) \t: %u KiB\n", deviceProp.totalConstMem/KILO);
	}
	// seleccion del dispositivo con mayor memoria global
	cudaSetDevice(SelectedDevice);
	printf("***************************************************\n");
	printf("Dispositivo seleccionado: DEVICE %d\n",SelectedDevice);
	cudaGetDeviceProperties(&deviceProp, SelectedDevice);
	printf("> Nombre : %s\n", deviceProp.name);
	printf("> Memoria: %u MiB\n", deviceProp.totalGlobalMem/MEGA);
	printf("> Hilos disponibles: %d \n", deviceProp.maxThreadsPerBlock);
	printf("***************************************************\n");
}

int siguiente_potencia2(int numero) {
	numero--;
	numero |= numero >> 1;
	numero |= numero >> 2;
	numero |= numero >> 4;
	numero |= numero >> 8;
	numero |= numero >> 16;
	numero++;
	return numero;
}

int clean_stdin(){
  while(getchar() != '\n');
  return 1;
}
