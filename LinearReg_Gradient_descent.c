#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ALPHA 0.01 // taux d'apprentissage
#define MAX_ITER 1000 // nombre maximum d'itérations
#define EPSILON 0.0001 // critère d'arrêt

// Structure pour stocker les données d'entraînement
typedef struct {
    double x; // variable indépendante
    double y; // variable dépendante
} TrainingExample;

// Fonction de prédiction
double predict(double x, double slope, double intercept) {
    return slope * x + intercept;
}

// Fonction de coût
double costFunction(TrainingExample *examples, int m, double slope, double intercept) {
    double cost = 0.0;
    for (int i = 0; i < m; i++) {
        double h = predict(examples[i].x, slope, intercept);
        cost += (h - examples[i].y) * (h - examples[i].y);
    }
    return cost / (2.0 * m);
}

// Fonction d'entraînement
void train(TrainingExample *examples, int m, double *slope, double *intercept) {
    double delta_slope, delta_intercept, cost, prev_cost = INFINITY;
    int iter = 0;
    do {
        delta_slope = 0.0;
        delta_intercept = 0.0;
        for (int i = 0; i < m; i++) {
            double h = predict(examples[i].x, *slope, *intercept);
            delta_slope += (h - examples[i].y) * examples[i].x;
            delta_intercept += h - examples[i].y;
        }
        delta_slope /= m;
        delta_intercept /= m;
        *slope -= ALPHA * delta_slope;
        *intercept -= ALPHA * delta_intercept;
        cost = costFunction(examples, m, *slope, *intercept);
        iter++;
    } while (iter < MAX_ITER && fabs(prev_cost - cost) > EPSILON);
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

    // Initialisation des paramètres
    double slope = 0.0, intercept = 0.0;

    // Entraînement du modèle
    train(examples, m, &slope, &intercept);

    // Affichage des résultats
    printf("Droite de régression : y = %fx + %f\n", slope, intercept);

    return 0;
}