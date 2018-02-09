#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>
#include <cuda_runtime.h>

#define MEGA (1<<20)
#define KILO (1<<10)

void dispositivos(int *, int *);
int clean_stdin();

__constant__ int dev_aleatorios[64][64];

__global__ void contar(int *resultado) {
	int posX = threadIdx.x;
	int posY = threadIdx.y;
	int izquierda = posX == 0 ? blockDim.x - 1 : posX - 1;
	int derecha = posX == blockDim.x - 1 ? 0 : posX + 1;
	int arriba = posY == 0 ? blockDim.y - 1 : posY - 1;
	int abajo = posY == blockDim.y - 1 ? 0 : posY + 1;
	resultado[posY * blockDim.x + posX] = dev_aleatorios[izquierda][posY] + dev_aleatorios[derecha][posY] + dev_aleatorios[posX][arriba] + dev_aleatorios[posX][abajo];
}

int main(int argc, char** argv) {
	SetConsoleTitle("ARCO - PRACTICA 7");
	int hilosMaximos;
	int dimension[3];
	int x = 0;
	dispositivos(&hilosMaximos, dimension);//, dimension);
	printf("\n\n> Dimensiones maximas: x=%d, y=%d, z=%d\n", dimension[0], dimension[1], dimension[2]);
	while (x <= 0) {
		printf("Introduce X: ");
		char enter;
		int res = scanf("%d%c", &x, &enter);
		if (res != 2 || enter != '\n'){
			x = 0;
			clean_stdin();
		}
	}
	int y = 0;
	while (y <= 0) {
		printf("Introduce Y: ");
		char enter;
		int res = scanf("%d%c", &y, &enter);
		if (res != 2 || enter != '\n'){
			y = 0;
			clean_stdin();
		}
	}

	int total_hilos = x*y;
	if (total_hilos > hilosMaximos) {
		printf("\n! No es posible lanzar tantos hilos\n");
		printf("\npulsa INTRO para finalizar..."); fflush(stdin);
		char tecla = getchar();
		return -1;
	}

	if (x < 3 || y < 3) printf("\n!> Al lanzar una dimensión de tamano inferior a 3, algunos de los elementos adyacentes de un hilo son el mismo.\n");
	printf("\n\n> Numero de hilos lanzados: %d\n", total_hilos);

	int NBloques = 1;
	dim3 NHilos(x,y);
	int *resultado, *dev_resultado;
	int aleatorios[64][64];

	
	for (int i = 0; i < x; i++)
		for (int j = 0; j < y; j++)
			aleatorios[i][j] = rand() % 2;

	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);
	cudaEventCreate(&stop);

	resultado = (int *)malloc(y*sizeof(aleatorios[0]));
	cudaMalloc( (void**)&dev_resultado, y*sizeof(aleatorios[0]));
	cudaMemcpyToSymbol(dev_aleatorios, aleatorios, y * sizeof(aleatorios[0]));

	contar<<<NBloques,NHilos>>>(dev_resultado);
	cudaMemcpy(resultado, dev_resultado, y*sizeof(aleatorios[0]), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree( dev_resultado );

	printf("\nAleatorios:\n");
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++)
			printf("%2d ", aleatorios[j][i]);
		printf("\n");
	}
	printf("\nResultado:\n");
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++)
			printf("%2d ", resultado[i*x + j]);
		printf("\n");
	}
	
	printf("\n\n> Tiempo de ejecucion: %f ms\n",elapsedTime);

	printf("\npulsa INTRO para finalizar..."); fflush(stdin);
	char tecla = getchar();
	return 0;
}

void dispositivos(int * hilosMaximos, int * dimension) {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
	printf("!!!!!No se han encontrado dispositivos CUDA!!!!!\n");
	hilosMaximos = 0;
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
	* hilosMaximos = deviceProp.maxThreadsPerBlock;
	memcpy(dimension, deviceProp.maxThreadsDim, sizeof(deviceProp.maxThreadsDim));
}

int clean_stdin(){
  while(getchar() != '\n');
  return 1;
}
