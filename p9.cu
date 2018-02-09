#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>
#include <cuda_runtime.h>
#include "cpu_bitmap.h"

#define MEGA (1<<20)
#define KILO (1<<10)

#define PIXELESPORBLOQUE (1<<7)
#define HILOSMAX 16

void dispositivos(int *, int *);
int clean_stdin();

__global__ void kernel(unsigned char *imagen, int ancho, int alto) {
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x < ancho && y < alto) {
		int pixel = x + y * ancho;
		unsigned char r = imagen[pixel *4 + 2], g = imagen[pixel *4 + 1], b = imagen[pixel *4 + 0];
		unsigned char gris = 0.299*r + 0.587*g + 0.114*b;
		imagen[pixel *4 + 0] = gris; // canal R
		imagen[pixel *4 + 1] = gris; // canal G
		imagen[pixel *4 + 2] = gris; // canal B
		imagen[pixel *4 + 3] = 255; // canal alfa
	}
}

int main(int argc, char** argv) {
	SetConsoleTitle("ARCO - PRACTICA 9");

	FILE * archivo;
	archivo = fopen("imagen.bmp", "rb");
	unsigned char * host_imagen_color, * dev_imagen;

	fseek(archivo, 0x0, SEEK_SET);
	char cabeceraBM[2];
	fread(cabeceraBM, 1, 2, archivo);
	printf("%c%c\n", cabeceraBM[0], cabeceraBM[1]);
	if (cabeceraBM[0] == 'B' && cabeceraBM[1] == 'M') {
		long comienzoDatos;
		fseek(archivo, 0x8, SEEK_CUR);
		fread(&comienzoDatos, 4, 1, archivo);
		printf("COMIENZO DATOS: %x\n", comienzoDatos);
		long ancho;
		fseek(archivo, 0x4, SEEK_CUR);
		fread(&ancho, 4, 1, archivo);
		printf("ANCHO: %d\n", ancho);
		long alto;
		fseek(archivo, 0, SEEK_CUR);
		fread(&alto, 4, 1, archivo);
		int tamañoTotal = alto*ancho;
		printf("ALTO: %d\nTOTAL: %d\n", alto, tamañoTotal);
		fseek(archivo, comienzoDatos, SEEK_SET);
		host_imagen_color = (unsigned char *)malloc(tamañoTotal*sizeof(unsigned char)*4);
		cudaMalloc( (void**)&dev_imagen, tamañoTotal*sizeof(unsigned char)*4);

		for (int i = 0; i < tamañoTotal; i++) {
			unsigned char color[3];
			for (int j = 0; j < 3; j++) {
				fread(&color[j], 1, 1, archivo);
				host_imagen_color[i*4+j] = color[j];
			}
			host_imagen_color[i*4+3] = 0;
		}
		int bloquesH = (ancho+HILOSMAX-1)/HILOSMAX;
		int bloquesV = (alto+HILOSMAX-1)/HILOSMAX;
		cudaMemcpy(dev_imagen, host_imagen_color, tamañoTotal*sizeof(unsigned char)*4, cudaMemcpyHostToDevice);
		dim3 Nbloques(bloquesH, bloquesV);
		dim3 hilosB(HILOSMAX, HILOSMAX);
		CPUBitmap bitmap(ancho, alto);
		unsigned char *host_bitmap = bitmap.get_ptr();

		kernel<<<Nbloques,hilosB>>>( dev_imagen, ancho, alto);

		cudaMemcpy( host_bitmap, dev_imagen, tamañoTotal*sizeof(unsigned char)*4, cudaMemcpyDeviceToHost );
		cudaFree( dev_imagen );

		bitmap.display_and_exit();
	}

	fclose(archivo);
	char enter;
	scanf("%c", &enter);
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
