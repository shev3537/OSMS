#include <stdio.h>
#include <math.h>

// Функция для вычисления скалярного произведения двух векторов
int Corr(int a[], int b[], int n) {
    int p = 0;
    for (int i = 0; i < n; i++) {
        p += a[i] * b[i];
    }
    return p;
}

// Функция для вычисления коэффициента корреляции между двумя векторами
double Corr2(int a[], int b[], int n) {
    double p = 0;
    double a_2 = 0;
    double b_2 = 0;

    // Вычисление произведения и квадратов векторов для корреляции
    for (int i = 0; i < n; i++) {
        p += (double) a[i] * b[i];
        a_2 += (double) a[i] * a[i];
        b_2 += (double) b[i] * b[i];
    }

    // Вычисление коэффициента корреляции
    double corr = p / (sqrt(a_2) * sqrt(b_2));

    return corr;
}

int main() {
    int a[] = {6, 2, 3, -2, -4, -4, 1, 1};
    int b[] = {3, 1, 5, 0, -3, -4, 2, 3};
    int c[] = {-4, -1, 3, -9, 2, -1, 4, -1};

    int n = sizeof(a) / sizeof(a[0]);

    // Вычисление корреляции между векторами a, b, c
    int ab = Corr(a, b, n);
    int bc = Corr(b, c, n);
    int ac = Corr(a, c, n);

    // Вывод матрицы коэффициентов корреляции между a, b, c
    printf("Correlation of a, b & c\n");
    printf("  |  a  |  b  |  c  |\n");
    printf(" a|  -  |%5.d|%5.d|\n", ab, ac);
    printf(" b|%5.d|  -  |%5.d|\n", ab, bc);
    printf(" c|%5.d|%5.d|  -  |\n", ac, bc);

    // Вычисление коэффициента корреляции между a, b, c
    float ab_2 = Corr2(a, b, n);
    float bc_2 = Corr2(b, c, n);
    float ac_2 = Corr2(a, c, n);

    // Вывод матрицы коэффициентов корреляции между a, b, c
    printf("Cross-correlation of a, b & c\n");
    printf("  |  a   |  b  |  c  |\n");
    printf(" a|   -  |%.2f |%.2f |\n", ab_2, ac_2);
    printf(" b| %.2f |  -  |%.2f|\n", ab_2, bc_2);
    printf(" c| %.2f |%.2f |  -  |\n", ac_2, bc_2);

    return 0;
}
