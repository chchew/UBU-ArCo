#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>
#include <cuda_runtime.h>

#define MEGA (1<<20)
#define KILO (1<<10)

void dispositivos();
int clean_stdin();

__constant__ int dev_aleatorios[1023];

__global__ void ordenar(int *resultado, int *rangos, int tamano) {
	int gId = threadIdx.x + blockDim.x * blockIdx.x;
	int rango = 0;
	bool lock = false;
	for (int i = 0; i < tamano; i++) {
		if (dev_aleatorios[gId] > dev_aleatorios[i] || (dev_aleatorios[gId] == dev_aleatorios[i] && gId != i && !lock)) {
			rango++;
		}
		else if (gId == i)
			lock = true;
	}
	rangos[gId] = rango;
	resultado[rango] = dev_aleatorios[gId];
}

int main(int argc, char** argv) {
	SetConsoleTitle("ARCO - PRACTICA 6");
	dispositivos();
	int bloques = 0;
	while (bloques <= 0) {
		printf("Introduce el numero de bloques: ");
		char enter;
		int res = scanf("%d%c", &bloques, &enter);
		if (res != 2 || enter != '\n'){
			bloques = 0;
			clean_stdin();
		}
	}
	int hilos = 0;
	while (hilos <= 0) {
		printf("Introduce el numero de hilos por bloque: ");
		char enter;
		int res = scanf("%d%c", &hilos, &enter);
		if (res != 2 || enter != '\n'){
			hilos = 0;
			clean_stdin();
		}
	}

	int total_hilos = bloques*hilos;
	printf("\n\n> Numero de hilos lanzados: %d\n", total_hilos);


	int *resultado, *aleatorios, *rangos;
	int *dev_resultado, *dev_rangos; // reserva en el host
	
	// declaracion de eventos
	cudaEvent_t start;
	cudaEvent_t stop;

	// creacion de eventos
	cudaEventCreate(&start);
	cudaEventRecord(start,0);
	cudaEventCreate(&stop);
	
	resultado = (int *)malloc(total_hilos*sizeof(int));
	aleatorios = (int *)malloc(total_hilos*sizeof(int));
	rangos = (int *)malloc(total_hilos*sizeof(int));
	
	// reserva en el device
	
	cudaMalloc( (void**)&dev_resultado, total_hilos*sizeof(int));
	cudaMalloc( (void**)&dev_rangos, total_hilos*sizeof(int));

	for (int i = 0; i < total_hilos; i++)
		aleatorios[i] = rand() % 60;

	cudaMemcpyToSymbol(dev_aleatorios, aleatorios, total_hilos * sizeof(int));
	
	ordenar<<<bloques, hilos>>>(dev_resultado, dev_rangos, total_hilos);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime,start,stop);
	
	cudaMemcpy(resultado, dev_resultado, total_hilos*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(rangos, dev_rangos, total_hilos*sizeof(int), cudaMemcpyDeviceToHost);

	printf("\nAleatorios:\n");
	for (int i = 0; i < total_hilos; i++)
		printf("%2d ", aleatorios[i]);
	
	printf("\n\nRangos:\n");

	for (int i = 0; i < total_hilos; i++)
		printf("%2d ", rangos[i]);

	printf("\n\nOrdenacion:\n");

	for (int i = 0; i < total_hilos; i++)
		printf("%2d ", resultado[i]);

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

int clean_stdin(){
  while(getchar() != '\n');
  return 1;
}
