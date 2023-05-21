#include <stdio.h>
#include <stdlib.h>

// Structure pour stocker les données d'entraînement
typedef struct {
    double x; // variable indépendante
    double y; // variable dépendante
} TrainingExample;

// Fonction pour calculer la droite de régression
void linearRegression(TrainingExample *examples, int m, double *slope, double *intercept) {
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
    for (int i = 0; i < m; i++) {
        sum_x += examples[i].x;
        sum_y += examples[i].y;
        sum_xy += examples[i].x * examples[i].y;
        sum_x2 += examples[i].x * examples[i].x;
    }
    double denominator = m * sum_x2 - sum_x * sum_x;
    *slope = (m * sum_xy - sum_x * sum_y) / denominator;
    *intercept = (sum_y - *slope * sum_x) / m;
}

int main() {
    // Données d'entraînement
    TrainingExample examples[] = {
        { 1.0, 3.0 },
        { 2.0, 5.0 },
        { 3.0, 7.0 },
        { 4.0, 9.0 },
        { 5.0, 11.0 }
    };
    int m = sizeof(examples) / sizeof(examples[0]); // nombre d'exemples

    // Calcul de la droite de régression
    double slope, intercept;
    linearRegression(examples, m, &slope, &intercept);

    // Affichage des résultats
    printf("Droite de régression : y = %fx + %f\n", slope, intercept);

    return 0;
}