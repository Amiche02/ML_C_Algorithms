#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ALPHA 0.01 // taux d'apprentissage
#define MAX_ITER 1000 // nombre maximum d'itérations
#define EPSILON 0.0001 // critère d'arrêt

// Structure pour stocker les données d'entraînement
typedef struct {
    double *x; // vecteur de caractéristiques
    int y; // variable cible (0 ou 1)
} TrainingExample;

// Fonction sigmoïde
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// Fonction de coût
double costFunction(double *theta, TrainingExample *examples, int m, int n) {
    double cost = 0.0;
    for (int i = 0; i < m; i++) {
        double h = sigmoid(theta[0] + theta[1] * examples[i].x[0] + theta[2] * examples[i].x[1]); // prédiction
        cost += -examples[i].y * log(h) - (1 - examples[i].y) * log(1 - h); // fonction de coût
    }
    cost /= m;
    return cost;
}

// Algorithme de descente de gradient
void gradientDescent(double *theta, TrainingExample *examples, int m, int n) {
    double h, gradient0, gradient1, gradient2;
    int iter = 0;
    double prevCost = costFunction(theta, examples, m, n);
    double newCost;

    while (iter < MAX_ITER) {
        gradient0 = 0.0;
        gradient1 = 0.0;
        gradient2 = 0.0;
        for (int i = 0; i < m; i++) {
            h = sigmoid(theta[0] + theta[1] * examples[i].x[0] + theta[2] * examples[i].x[1]);
            gradient0 += h - examples[i].y;
            gradient1 += (h - examples[i].y) * examples[i].x[0];
            gradient2 += (h - examples[i].y) * examples[i].x[1];
        }
        gradient0 /= m;
        gradient1 /= m;
        gradient2 /= m;
        theta[0] -= ALPHA * gradient0;
        theta[1] -= ALPHA * gradient1;
        theta[2] -= ALPHA * gradient2;

        newCost = costFunction(theta, examples, m, n);

        if (fabs(newCost - prevCost) < EPSILON) {
            break;
        }

        prevCost = newCost;
        iter++;
    }
}

int main() {
    // Données d'entraînement
    TrainingExample examples[] = {
        { (double[]) { 1.0, 2.0 }, 0 },
        { (double[]) { 2.0, 1.0 }, 0 },
        { (double[]) { 3.0, 4.0 }, 0 },
        { (double[]) { 4.0, 3.0 }, 1 },
        { (double[]) { 5.0, 6.0 }, 1 },
        { (double[]) { 6.0, 5.0 }, 1 }
    };
    int m = sizeof(examples) / sizeof(examples[0]); // nombre d'exemples
    int n = 2; // nombre de caractéristiques

    // Initialisation des paramètres
    double theta[] = { 0.0, 0.0, 0.0 };

    // Entraînement du modèle
    gradientDescent(theta, examples, m, n);

    // Prédiction sur de nouvelles données
    double x1 = 7.0;
    double x2 = 8.0;
    double h = sigmoid(theta[0] + theta[1] * x1 + theta[2] * x2);
    printf("Prédiction pour x1 = %f, x2 = %f : %f\n", x1, x2, h);

    return 0;
}