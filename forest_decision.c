#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_FEATURES 20 // nombre maximal de caractéristiques dans les exemples
#define MAX_DEPTH 10 // profondeur maximale de l'arbre de décision

// Structure de données pour représenter les nœuds de l'arbre de décision
typedef struct TreeNode {
    int is_leaf; // indique si le nœud est une feuille
    int label; // étiquette de classe prédite pour les exemples qui atteignent ce nœud
    int split_feature; // indice de la caractéristique utilisée pour la division
    float split_value; // valeur de la caractéristique utilisée pour la division
    struct TreeNode* left_child; // sous-arbre pour les exemples avec des valeurs inférieures ou égales à la valeur de la caractéristique de division
    struct TreeNode* right_child; // sous-arbre pour les exemples avec des valeurs supérieures à la valeur de la caractéristique de division
} TreeNode;

// Génère des exemples de données aléatoires
float** create_random_features(int num_examples, int num_features);

// Génère des étiquettes de classe aléatoires pour les exemples de données
int* create_random_labels(int num_examples);

// Calcule l'entropie d'un ensemble d'exemples
float calculate_entropy(int* labels, int num_examples);

// Divise un ensemble d'exemples en deux sous-ensembles en fonction d'une caractéristique et d'une valeur de division
int* split_examples(float** features, int* labels, int num_examples, int split_feature, float split_value, int* num_left, int* num_right);

// Trouve la caractéristique et la valeur de division optimales pour diviser un ensemble d'exemples
void find_best_split(float** features, int* labels, int num_examples, int num_features, int* best_feature, float* best_value);

// Construit un arbre de décision en utilisant la récursion
TreeNode* build_decision_tree(float** features, int* labels, int num_examples, int num_features, int depth);

// Prédit l'étiquette de classe pour un exemple de données donné en utilisant un arbre de décision
int predict(TreeNode* node, float* example);

int main() {
    int num_examples = 1000;
    int num_features = 10;
    float** features = create_random_features(num_examples, num_features);
    int* labels = create_random_labels(num_examples);

    TreeNode* root = build_decision_tree(features, labels, num_examples, num_features, 0);

    // Exemple de prédiction pour un nouvel exemple
    float* example = create_random_features(1, num_features)[0];
    int predicted_label = predict(root, example);
    printf("Predicted label: %d\n", predicted_label);

    return 0;
}

float** create_random_features(int num_examples, int num_features) {
    float** features = (float**) malloc(num_examples * sizeof(float*));
    for (int i = 0; i < num_examples; i++) {
        features[i] = (float*) malloc(num_features * sizeof(float));
        for (int j = 0; j < num_features; j++) {
            features[i][j] = (float) rand() / RAND_MAX;
        }
    }
    return features;
}

int* create_random_labels(int num_examples) {
    int* labels = (int*) malloc(num_examples * sizeof(int));
    for (int i = 0; i < num_examples; i++) {
        labels[i] = rand() % 2;
    }
    return labels;
}

float calculate_entropy(int* labels, int num_examples) {
    int num_positives = 0;
    for (int i = 0; i < num_examples; i++) {
        if (labels[i] == 1) {
            num_positives++;
        }
    }
    float p_positives = (float) num_positives / num_examples;
    float p_negatives = 1 - p_positives;
    float entropy = 0;
    if (p_positives > 0) {
        entropy -= p_positives * log2f(p_positives);
    }
    if (p_negatives > 0) {
        entropy -= p_negatives * log2f(p_negatives);
    }
    return entropy;
}

int** split_examples(float** features, int* labels, int num_examples, int split_feature, float split_value, int* num_left, int* num_right)    int* left_indices = (int*) malloc(num_examples * sizeof(int));
    int* right_indices = (int*) malloc(num_examples * sizeof(int));
    int num_left_examples = 0;
    int num_right_examples = 0;
    for (int i = 0; i < num_examples; i++) {
        if (features[i][split_feature] <= split_value) {
            left_indices[num_left_examples] = i;
            num_left_examples++;
        } else {
            right_indices[num_right_examples] = i;
            num_right_examples++;
        }
    }
    *num_left = num_left_examples;
    *num_right = num_right_examples;
    int* left_labels = (int*) malloc(num_left_examples * sizeof(int));
    int* right_labels = (int*) malloc(num_right_examples * sizeof(int));
    for (int i = 0; i < num_left_examples; i++) {
        left_labels[i] = labels[left_indices[i]];
    }
    for (int i = 0; i < num_right_examples; i++) {
        right_labels[i] = labels[right_indices[i]];
    }
    free(left_indices);
    free(right_indices);
    int** result = (int**) malloc(2 * sizeof(int*));
    result[0] = left_labels;
    result[1] = right_labels;
    return result;
}

void find_best_split(float** features, int* labels, int num_examples, int num_features, int* best_feature, float* best_value) {
    float best_info_gain = -1;
    for (int i = 0; i < num_features; i++) {
        for (int j = 0; j < num_examples; j++) {
            float split_value = features[j][i];
            int num_left = 0;
            int num_right = 0;
            int* child_labels = split_examples(features, labels, num_examples, i, split_value, &num_left, &num_right);
            float p_left = (float) num_left / num_examples;
            float p_right = (float) num_right / num_examples;
            float info_gain = calculate_entropy(labels, num_examples) - p_left * calculate_entropy(child_labels[0], num_left) - p_right * calculate_entropy(child_labels[1], num_right);
            if (info_gain > best_info_gain) {
                best_info_gain = info_gain;
                *best_feature = i;
                *best_value = split_value;
            }
            free(child_labels[0]);
            free(child_labels[1]);
            free(child_labels);
        }
    }
}

TreeNode* build_decision_tree(float** features, int* labels, int num_examples, int num_features, int depth) {
    TreeNode* node = (TreeNode*) malloc(sizeof(TreeNode));
    node->is_leaf = 0;
    node->label = -1;
    node->split_feature = -1;
    node->split_value = -1;
    node->left_child = NULL;
    node->right_child = NULL;
    if (depth == MAX_DEPTH || num_examples == 0) {
        node->is_leaf = 1;
        if (num_examples == 0) {
            node->label = -1;
        } else {
            int num_positives = 0;
            for (int i = 0; i < num_examples; i++) {
                if (labels[i] == 1) {
                    num_positives++;
                }
            }
            if (num_positives >= num_examples - num_positives) {
                node->label = 1;
            } else {
                node->label = 0;
            }
        }
    } else {
        int best_feature;
        float best_value;
        find_best_split(features, labels, num_examples, num_features, &best_feature, &best_value);
        int num_left = 0;
        int num_right = 0;
        int* child_labels = split_examples(features, labels, num_examples, best_feature, best_value, &num_left, &num_right);
        node->split_feature = best_feature;
        node->split_value = best_value;
        node->left_child = build_decision_tree(features, child_labels[0], num_left, num_features, depth + 1);
        node->right_child = build_decision_tree(features, child_labels[1], num_right, num_features, depth + 1);
        free(child_labels[0]);
        free(child_labels[1]);
        free(child_labels);
    }
    return node;
}

int predict(TreeNode* node, float* example) {
    if (node->is_leaf) {
        return node->label;
    } else {
        if (example[node->split_feature] <= node->split_value) {
            return predict(node->left_child, example);
        } else {
            return predict(node->right_child, example);
        }
    }
}