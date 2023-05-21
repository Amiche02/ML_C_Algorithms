#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_DEPTH 10 // profondeur maximale de l'arbre
#define MIN_EXAMPLES 5 // nombre minimal d'exemples pour une feuille

// Structure pour stocker les données d'entraînement
typedef struct {
    double *features; // vecteur de caractéristiques
    int label; // étiquette de classe (0 ou 1)
} TrainingExample;

// Structure pour représenter un nœud de l'arbre
typedef struct node {
    struct node *left; // sous-arbre gauche
    struct node *right; // sous-arbre droit
    int feature_index; // indice de la caractéristique utilisée pour la division
    double threshold; // valeur de seuil pour la division
    int label; // étiquette de classe (0 ou 1) pour les feuilles
} Node;

// Fonction pour calculer l'entropie de Shannon
double entropy(int *counts, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        if (counts[i] > 0) {
            double p = (double)counts[i] / (double)n;
            total -= p * log2(p);
        }
    }
    return total;
}

// Fonction pour diviser les exemples selon une caractéristique et une valeur de seuil
void splitExamples(TrainingExample *examples, int m, int feature_index, double threshold, TrainingExample **left, int *left_count, TrainingExample **right, int *right_count) {
    *left = calloc(m, sizeof(TrainingExample));
    *right = calloc(m, sizeof(TrainingExample));
    *left_count = 0;
    *right_count = 0;
    for (int i = 0; i < m; i++) {
        if (examples[i].features[feature_index] <= threshold) {
            (*left)[*left_count] = examples[i];
            (*left_count)++;
        } else {
            (*right)[*right_count] = examples[i];
            (*right_count)++;
        }
    }
}

// Fonction pour trouver la meilleure division d'un ensemble d'exemples
void findBestSplit(TrainingExample *examples, int m, int n, int *used_features, int used_count, int *best_feature_index, double *best_threshold, double *best_score) {
    int class_counts[2] = {0, 0};
    for (int i = 0; i < m; i++) {
        class_counts[examples[i].label]++;
    }
    double base_entropy = entropy(class_counts, m);
    *best_score = 0.0;
    for (int i = 0; i < n; i++) {
        if (used_features[i]) {
            continue;
        }
        double thresholds[MIN_EXAMPLES];
        int threshold_count = 0;
        for (int j = 0; j < m; j++) {
            if (threshold_count < MIN_EXAMPLES || examples[j].features[i] != thresholds[threshold_count - 1]) {
                thresholds[threshold_count] = examples[j].features[i];
                threshold_count++;
            }
        }
        for (int j = 0; j < threshold_count; j++) {
            TrainingExample *left, *right;
            int left_count, right_count;
            splitExamples(examples, m, i, thresholds[j], &left, &left_count, &right, &right_count);
            if (left_count > 0 && right_count > 0) {
                int left_class_counts[2] = {0, 0};
                int right_class_counts[2] = {0, 0};
                for (int k = 0; k < left_count; k++) {
                    left_class_counts[left[k].label]++;
                }
                for (int k = 0; k < right_count; k++) {
                    right_class_counts[right[k].label]++;
                }
                double left_entropy = entropy(left_class_counts, left_count);
                double right_entropy = entropy(right_class_counts, right_count);
                double info_gain = base_entropy - ((double)left_count / (double)m) * left_entropy - ((double)right_count / (double)m) * right_entropy;
                if (info_gain > *best_score) {
                    *best_feature_index = i;
                    *best_threshold = thresholds[j];
                    *best_score = info_gain;
                }
                free(left);
                free(right);
            }
        }
    }
}

// Fonction pour construire un arbre de décision à partir d'un ensemble d'exemples
Node *buildTree(TrainingExample *examples, int m, int n, int depth) {
    int class_counts[2] = {0, 0};
    for (int i = 0; i < m; i++) {
        class_counts[examples[i].label]++;
    }
    if (class_counts[0] == m || class_counts[1] == m || depth >= MAX_DEPTH) {
        Node *leaf = malloc(sizeof(Node));
        leaf->left = NULL;
        leaf->right = NULL;
        leaf->feature_index = -1;
        leaf->threshold = 0.0;
        if (class_counts[0] >= class_counts[1]) {
            leaf->label = 0;
        } else {
            leaf->label = 1;
        }
        return leaf;
    }
    int *used_features = calloc(n, sizeof(int));
    int used_count = 0;
    Node *node = malloc(sizeof(Node));
    node->feature_index = -1;
    node->threshold = 0.0;
    node->label = -1;
    double best_score = 0.0;
    findBestSplit(examples, m, n, used_features, used_count, &node->feature_index, &node->threshold, &best_score);
    free(used_features);
    if (node->feature_index == -1) {
        Node *leaf = malloc(sizeof(Node));
        leaf->left = NULL;
        leaf->right = NULL;
        leaf->feature_index = -1;
        leaf->threshold = 0.0;
        if (class_counts[0] >= class_counts[1]) {
            leaf->label = 0;
        } else {
            leaf->label = 1;
        }
        return leaf;
    }
    TrainingExample *left, *right;
    int left_count, right_count;
    splitExamples(examples, m, node->feature_index, node->threshold, &left, &left_count, &right, &right_count);
    node->left = buildTree(left, left_count, n, depth + 1);
    node->right = buildTree(right, right_count, n, depth + 1);
    free(left);
    free(right);
    return node;
}

// Fonction pour prédire l'étiquette de classe d'un exemple en utilisant un arbre de décision
int predict(Node *root, double *features) {
    while (root->feature_index != -1) {
        if (features[root->feature_index] <= root->threshold) {
            root = root->left;
        } else {
            root = root->right;
        }
    }
    return root->label;
}

// Exemple d'utilisation
int main() {
    // Exemples d'entraînement
    TrainingExample examples[] = {
        {(double[]) {2.0, 3.0}, 0},
        {(double[]) {4.0, 2.0}, 0},
        {(double[]) {4.0, 4.0}, 1},
        {(double[]) {6.0, 2.0}, 1},
        {(double[]) {7.0, 4.0}, 1},
        {(double[]) {8.0, 1.0}, 0},
        {(double[]) {9.0, 3.0}, 1},
        {(double[]) {10.0, 2.0}, 0},
        {(double[]) {11.0, 4.0}, 1},
        {(double[]) {12.0, 3.0}, 1}
    };
    int m = sizeof(examples) / sizeof(TrainingExample);
    int n = sizeof(examples[0].features) / sizeof(double);

    // Construction de l'arbre de décision
    Node *root = buildTree(examples, m, n, 0);

    // Prédiction de l'étiquette de classe pour un exemple
    double features[] = {5.0, 3.0};
    int label = predict(root, features);
    printf("Label: %d\n", label);

    // Libération de la mémoire
    free(root);

    return 0;
}