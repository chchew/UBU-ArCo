#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>
#include <time.h>
#include <cuda_runtime.h>

__host__ void setEvent(event *ev);

#ifdef __linux__
	#include <sys/time.h>
	typedef struct timeval event;
	gettimeofday(ev, NULL);
#else
	#include <windows.h>
	typedef LARGE_INTEGER event;
	QueryPerformanceCounter(ev);
#endif

#define MEGA (1<<20)
#define KILO (1<<10)

#define CICLOS 10

void dispositivos(int *, int *);
void rellenarAleatorios(int *, int);
int clean_stdin();

__global__ void ordenar(int * dev_aleatorios,int *resultado, int *rangos, int tamano) {
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
	SetConsoleTitle("ARCO - PRACTICA 10");
	int hilosMaximos;
	int dimension[3];
	dispositivos(&hilosMaximos, dimension);

	//printf("%d - %d %d %d", hilosMaximos, dimension[0], dimension[1], dimension[2]);

	int *resultado, *aleatorios, *rangos;
	int *dev_aleatorios, *dev_resultado, *dev_rangos; // reserva en el host
	float *tiempoGPU, *tiempoCPU;
	
	// declaracion de eventos
	cudaEvent_t start;
	cudaEvent_t stop;


	
	tiempoGPU = (float *)malloc(8*sizeof(float));
	tiempoCPU = (float *)malloc(8*sizeof(float));
	
	// reserva en el device
	
	cudaEventCreate(&start);
	cudaEventRecord(start,0);
	cudaEventCreate(&stop);

	for (int j = 0; j < 8; j ++) {
		int tamano = (1<<(j+5));
		aleatorios = (int *)malloc(tamano*sizeof(int));
		resultado = (int *)malloc(tamano*sizeof(int));
		cudaMalloc( (void**)&dev_aleatorios, tamano*sizeof(int));
		cudaMalloc( (void**)&dev_rangos, tamano*sizeof(int));
		cudaMalloc( (void**)&dev_resultado, tamano*sizeof(int));
		float tiempoTotal = 0;
		for (int i = 0; i < CICLOS; i++) {
			rellenarAleatorios(aleatorios, tamano);
			cudaMemcpy(dev_aleatorios, aleatorios, tamano * sizeof(int), cudaMemcpyHostToDevice);
			ordenar<<<tamano/hilosMaximos, hilosMaximos>>>(dev_aleatorios, dev_resultado, dev_rangos, tamano);
			cudaMemcpy(resultado, dev_resultado, tamano*sizeof(int), cudaMemcpyDeviceToHost);
			cudaEventRecord(stop,0);
			cudaEventSynchronize(stop);
			float elapsedTime;
			cudaEventElapsedTime(&elapsedTime,start,stop);
			tiempoTotal += elapsedTime;
		}
		tiempoGPU[j] = tiempoTotal / CICLOS;
		printf("\n\n> Tiempo de ejecucion de %d en gpu: %f ms\n", tamano, tiempoGPU[j]);
	}
	printf("\n");

	time_t inicioCPU;
	time_t finCPU;

	for (int j = 0; j < 8; j ++) {
		int tamano = (1<<(j+5));
		aleatorios = (int *)malloc(tamano*sizeof(int));
		rangos = (int *)malloc(tamano*sizeof(int));
		resultado = (int *)malloc(tamano*sizeof(int));
		float tiempoTotal = 0;
		for (int i = 0; i < CICLOS; i++) {
			event start, stop;
			setEvent(&start);

			int rango = 0;
			bool lock = false;
			for (int k = 0; k < tamano; k++) {
				if (aleatorios[i] > aleatorios[k] || (aleatorios[i] == aleatorios[k] && i != k && !lock)) {
					rango++;
				}
				else if (i == k)
					lock = true;
			}
			rangos[i] = rango;
			resultado[rango] = aleatorios[i];

			setEvent(&stop);
			tiempoTotal += eventDiff(&start, &stop);
		}
		tiempoCPU[j] = tiempoTotal / CICLOS;
		printf("\n> Tiempo de ejecucion de %d en cpu: %f ms\n", tamano, tiempoCPU[j]);
	}


	/*
	printf("|   N  |  32  |  64  |  128  |  256  |  512  |  1024  |  2048  |  4096  |\n");
	printf("| tCPU | %4f | %4f | %5f | %5f | %5f | %6f | %6f | %6f |\n", tiempoCPU[0], tiempoCPU[1], tiempoCPU[2], tiempoCPU[3], tiempoCPU[4], tiempoCPU[5], tiempoCPU[6], tiempoCPU[7]);
	printf("| tGPU | %4f | %4f | %5f | %5f | %5f | %6f | %6f | %6f |\n", tiempoGPU[0], tiempoGPU[1], tiempoGPU[2], tiempoGPU[3], tiempoGPU[4], tiempoGPU[5], tiempoGPU[6], tiempoGPU[7]);
	*/



	// liberacion de recursos
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree( dev_resultado );

	// salida
	printf("\npulsa INTRO para finalizar...");
	fflush(stdin);
	getchar();
	return 0;
}

void rellenarAleatorios(int * aleatorios, int cantidad) {
	for (int i = 0; i < cantidad; i++)
		aleatorios[i] = rand() % 51;
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

__host__ double eventDiff(event *first, event *last) {
	#ifdef __linux__
		return ((double)(last->tv_sec + (double)last->tv_usec/1000000)-
			 (double)(first->tv_sec + (double)first->tv_usec/1000000))*1000.0;
	#else
		event freq;
		QueryPerformanceFrequency(&freq);
		return ((double)(last->QuadPart - first->QuadPart) / (double)freq.QuadPart) * 1000.0;
	#endif
}
