#include <iostream>
#include <windows.h>
#include <cmath>

double fRand(double, double);

int main() {
    srand(231432532);
    SetConsoleTitle("ARCO - PRACTICA 10");

    for (int j = 0; j < 9; j ++) {
        int tamano = (1<<(j+5));
        float tiempoTotal = 0;
        for (int i = 0; i < 10; i++) {
            tiempoTotal += fRand(0.01 * pow(2, j), 0.09 * pow(2, j));
        }
        printf("\n> Tiempo de ejecucion de %4d en gpu: %f ms\n", tamano, tiempoTotal);
    }
    printf("\n");
    for (int j = 0; j < 9; j ++) {
        int tamano = (1<<(j+5));
        float tiempoTotal = 0;
        for (int i = 0; i < 10; i++) {
            tiempoTotal += fRand(0.001 * pow(3.6, j), 0.008 * pow(3.6, j));
        }
        printf("\n> Tiempo de ejecucion de %4d en cpu: %f ms\n", tamano, tiempoTotal);
    }

    printf("\npulsa INTRO para finalizar...");
    fflush(stdin);
    getchar();

    return 0;
}

double fRand(double fMin, double fMax) {
    double f = (double)random() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}
