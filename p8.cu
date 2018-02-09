#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>
#include <cuda_runtime.h>
#include "cpu_bitmap.h"

#define MEGA (1<<20)
#define KILO (1<<10)

#define DIM 8
#define PIXELESPORBLOQUE (1<<7)
#define HILOSMAX 16

void dispositivos(int *, int *);
int clean_stdin();

__global__ void kernel(unsigned char *imagen) {
	int bX = blockIdx.x / (PIXELESPORBLOQUE / HILOSMAX);
	int bY = blockIdx.y / (PIXELESPORBLOQUE / HILOSMAX);

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	// coordenada vertical
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	// coordenada global de cada pixel
	int pixel = x + y * blockDim.x * gridDim.x;
	// cada hilo pinta un pixel con un color arbitrario
	unsigned char color = 0;
	if ((bX + bY) % 2)
		color = 255;
	imagen[pixel *4 + 0] = color; // canal R
	imagen[pixel *4 + 1] = color; // canal G
	imagen[pixel *4 + 2] = color; // canal B
	imagen[pixel *4 + 3] = 255; // canal alfa
}

int main(int argc, char** argv) {
	SetConsoleTitle("ARCO - PRACTICA 8");
	int hilosMaximos;
	int dimension[3];
	dispositivos(&hilosMaximos, dimension);
	// declaracion del bitmap
	CPUBitmap bitmap( DIM*PIXELESPORBLOQUE, DIM*PIXELESPORBLOQUE );
	// tamaño en bytes
	size_t size = bitmap.image_size();
	// reserva en el host
	unsigned char *host_bitmap = bitmap.get_ptr();
	// reserva en el device
	unsigned char *dev_bitmap; cudaMalloc( (void**)&dev_bitmap, size );
	// generamos el bitmap
	dim3 Nbloques(DIM*PIXELESPORBLOQUE/HILOSMAX, DIM*PIXELESPORBLOQUE/HILOSMAX);
	dim3 hilosB(HILOSMAX, HILOSMAX);
	kernel<<<Nbloques,hilosB>>>( dev_bitmap );
	// recogemos el bitmap desde la GPU para visualizarlo
	cudaMemcpy( host_bitmap, dev_bitmap, size, cudaMemcpyDeviceToHost );
	// liberacion de recursos
	cudaFree( dev_bitmap );
	// visualizacion y salida
	bitmap.display_and_exit();
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
