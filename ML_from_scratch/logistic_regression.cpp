// name: aditi chaudhari
// assignment: ml from scratch
// class: cs 4375.003
// professor: dr. karen mazidi

#include <iostream>
#include <fstream>
#include <vector>
#include <valarray>
#include <chrono>

using namespace std;

vector<vector<double>> matrix_multiplication(const vector<vector<double>> &vec1, const vector<vector<double>> &vec2) {
    int vec1_m = vec1.size();
    int vec1_n = vec1[0].size();

    int vec2_m = vec2.size();
    int vec2_n = vec2[0].size();

    vector<vector<double>> result(vec1_m, vector<double> (vec2_n, 0));
    if (vec1_n != vec2_m) {
        cout << "matrix sizes not compatible" << endl;
        exit(0);
    }
    else{
        for (int j = 0; j < vec2_n; j++) {
            for (int k = 0; k < vec1_n; k++) {
                for (int i = 0; i < vec1_m; i++){
                    result[i][j] += vec1[i][k] * vec2[k][j];
                }
            }
        }
    }
    return result;
}

vector<vector<double>> matrix_subtraction(const vector<vector<double>> &vec1, const vector<vector<double>> &vec2) {
    vector<vector<double>> result(vec1.size(), vector<double> (vec1[0].size(), 0));
    if (vec1.size() == vec2.size() && vec1[0].size() == vec2[0].size()) {
        for (int i = 0; i < vec1.size(); i++) {
            for (int j = 0; j < vec1[0].size(); j++) {
                result[i][j] = vec1[i][j] - vec2[i][j];
            }
        }
    }
    return result;
}

vector<vector<double>> transpose_matrix(vector<vector<double>> &vec) {
    vector<vector<double>> transpose(vec[0].size(), vector<double>());

    for(int i = 0; i < vec.size(); i++) {
        for (int j = 0; j < vec[0].size(); j++) {
            transpose[j].push_back(vec[i][j]);
        }
    }
    return transpose;
}

vector<vector<double>> sigmoid(vector<vector<double>> &vec) {
    vector<vector<double>> sigmoid_values(vec.size(), vector<double>(1,0));

    for(int i = 0; i < vec.size(); i++) {
        sigmoid_values[i][0] = 1.0 / (1 + exp(-vec[i][0]));
    }

    return sigmoid_values;
}

vector<vector<double>> glm(vector<double> &sex, vector<double> &survived) {
    // r instruction: weights <- c(1, 1)
    vector<vector<double>> weights{
            {1},
            {1}
    };

    // r instruction: data_matrix <- cbind(rep(1, nrow(train)), train$sex)
    vector<vector<double>> data_matrix(sex.size(), vector<double>(2, 0));

    for(int i = 0; i < sex.size(); i++) {
        data_matrix[i][0] = 1;
        data_matrix[i][1] = sex[i];
    }

    // r instruction: labels <- as.double(train$survived)
    vector<vector<double>> labels(survived.size(), vector<double>(1, 0));

    for(int i = 0; i < survived.size(); i++) {
        labels[i][0] = survived.at(i);
    }

    // iteration (decided to iterate 5000 times as opposed to 50,000 times bc they return similar results)
    double learning_rate = 0.001;
    for (int i = 0; i < 5000; i++) {
        // r instruction: prob_vector <- sigmoid(data_matrix %*% weights)
        vector<vector<double>> matrix_mul = matrix_multiplication(data_matrix, weights);
        vector<vector<double>> prob_vector = sigmoid(matrix_mul);

        // r instruction: error <- labels - prob_vector
        vector<vector<double>> error = matrix_subtraction(labels, prob_vector);
        vector<vector<double>> transpose = transpose_matrix(data_matrix);

        // r instruction: weights <- weights + learning_rate * t(data_matrix) %*% error
        matrix_mul = matrix_multiplication(transpose, error);
        weights[0][0] += (learning_rate * matrix_mul[0][0]);
        weights[1][0] += (learning_rate * matrix_mul[1][0]);
    }
    return weights;
}

vector<double> predict(vector<double> &sex, double w0, double w1) {
    // if probability > 0.5, result[i] stores 1
    // else result[i] stores 0
    vector<double> result(sex.size());

    for (int i = 0; i < sex.size(); i++) {
        // calculate probability
        double log_odds = (w1 * sex[i]) + w0;
        double odds = exp(log_odds);
        double probability = odds / (1 + odds);

        if (probability > 0.50) {
            result[i] = 1;
        }
        else {
            result[i] = 0;
        }
    }
    return result;
}

vector<double> get_confusion_matrix_elements(vector<double> sex, vector<double> survived){
    vector<double> TP_FP_TN_FN(4);

    int TP = 0;
    int FP = 0;
    int TN = 0;
    int FN = 0;

    for (int i = 0; i < sex.size(); i++) {
        if (sex[i] == 1 and survived[i] == 1) {
            TP++;
        }
        else if (sex[i] == 1 and survived[i] == 0) {
            FP++;
        }
        else if (sex[i] == 0 and survived[i] == 1) {
            FN++;
        }
        else {
            TN++;
        }
    }

    TP_FP_TN_FN[0] = TP;
    TP_FP_TN_FN[1] = FP;
    TP_FP_TN_FN[2] = TN;
    TP_FP_TN_FN[3] = FN;

    return TP_FP_TN_FN;
}

int main() {
    // reading data from the titanic_project.csv file
    ifstream inFS;
    string line;

    string passenger_no_in, pclass_in, survived_in, sex_in, age_in;
    const int MAX_LENGTH = 1500;

    vector<double> pclass(MAX_LENGTH);
    vector<double> survived(MAX_LENGTH);
    vector<double> sex(MAX_LENGTH);
    vector<double> age(MAX_LENGTH);

    cout << "Building a Logistic Regression Model for the data in the titanic_project.csv file..." << endl;

    inFS.open("titanic_project.csv");

    if(!inFS.is_open()) {
        cout << "Could not open titanic_project.csv file" << endl;
        return 1;
    }

    getline(inFS, line);

    int numObservations = 0;

    while(inFS.good()) {
        getline(inFS, passenger_no_in, ',');
        getline(inFS, pclass_in, ',');
        getline(inFS, survived_in, ',');
        getline(inFS,sex_in, ',');
        getline(inFS, age_in, '\n');

        pclass.at(numObservations) = stof(pclass_in);
        survived.at(numObservations) = stof(survived_in);
        sex.at(numObservations) = stof(sex_in);
        age.at(numObservations) = stof(age_in);

        numObservations++;
    }

    pclass.resize(numObservations);
    survived.resize(numObservations);
    sex.resize(numObservations);
    age.resize(numObservations);

    inFS.close();

    // dividing into test and train data
    vector<double> train_sex(800);
    vector<double> test_sex(numObservations - 800);

    vector<double> train_survived(800);
    vector<double> test_survived(numObservations - 800);

    for(int i = 0; i < numObservations; i++){
        if (i < 800) {
            train_sex.at(i) = sex.at(i);
            train_survived.at(i) = survived.at(i);
        }
        else {
            test_sex.at(i - 800) = sex.at(i);
            test_survived.at( i - 800) = survived.at(i);
        }
    }

    // creating a logistic regression model with training data
    // and outputting the coefficients

    auto start = chrono::steady_clock::now();
    vector<vector<double>> coefficients = glm(train_sex, train_survived);
    auto end = chrono::steady_clock::now();

    cout << "Time taken to train the model with the training data: " << chrono::duration_cast<chrono::seconds>(end - start).count() << " seconds." << endl;
    cout << endl;

    cout << "Coefficient Outputs of the Logistic Regression Model: " << endl;
    double w0 = coefficients[0][0]; // intercept
    double w1 = coefficients[1][0]; // slope
    cout << "w0 (intercept): " << w0 << " , w1 (slope): " << w1 << endl;

    // evaluating the model with the test data
    test_sex = predict(test_sex, w0, w1);

    // calculating & then outputting accuracy, sensitivity, and specificity

    // confusion_matrix[0] = True Positive
    // confusion_matrix[1] = False Positive
    // confusion_matrix[2] = True Negative
    // confusion_matrix[3] = False Negative
    vector<double> TP_FP_TN_FN = get_confusion_matrix_elements(test_sex, test_survived);
    double TP = TP_FP_TN_FN[0];
    double FP = TP_FP_TN_FN[1];
    double TN = TP_FP_TN_FN[2];
    double FN = TP_FP_TN_FN[3];

    double accuracy = (TP + TN) / (TP + TN + FP + FN);
    double sensitivity = TP / (TP + FN);
    double specificity = TN / (TN + FP);

    cout << endl;
    cout << "Metrics: " << endl;
    cout << "accuracy: " << accuracy << endl;
    cout << "sensitivity: " << sensitivity << endl;
    cout << "specificity: " << specificity << endl;

    return 0;
}
