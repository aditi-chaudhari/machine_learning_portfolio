// name: aditi chaudhari
// assignment: ml from scratch
// class: cs 4375.003
// professor: dr. karen mazidi

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#define _USE_MATH_DEFINES
#include <math.h>
#include <iomanip>

using namespace std;

// function to calculate conditional probability from mean and variance
double calc_age_lh(double z, double mean, double var) {
    return 1.0 / sqrt(2.0 * M_PI * var) * exp(-(pow((z - mean),2)) / (2.0 * var));
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

    cout << "Building a Naive Bayes Model for the data in the titanic_project.csv file..." << endl;

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
    vector<double> train_pclass(800);
    vector<double> test_pclass(numObservations - 800);

    vector<double> train_survived(800);
    vector<double> test_survived(numObservations - 800);

    vector<double> train_sex(800);
    vector<double> test_sex(numObservations - 800);

    vector<double> train_age(800);
    vector<double> test_age(numObservations - 800);

    for(int i = 0; i < numObservations; i++){
        if (i < 800) {
            train_pclass.at(i) = pclass.at(i);
            train_survived.at(i) = survived.at(i);
            train_sex.at(i) = sex.at(i);
            train_age.at(i) = age.at(i);
        }
        else {
            test_pclass.at(i - 800) = pclass.at(i);
            test_survived.at( i - 800) = survived.at(i);
            test_sex.at(i - 800) = sex.at(i);
            test_age.at(i - 800) = age.at(i);
        }
    }

    auto start = chrono::steady_clock::now();

    // finding apriori probabilities
    int survived_no_count = 0;
    int survived_yes_count = 0;

    for(int i = 0; i < train_survived.size(); i++) {
       if (train_survived[i] == 0) {
           survived_no_count++;
       }
       else{
           survived_yes_count++;
       }
    }

    double apriori_survived_no = (double)survived_no_count / train_survived.size();
    double apriori_survived_yes = (double)survived_yes_count / train_survived.size();


    cout << endl;
    cout << "A-priori probabilities" << endl;
    cout << "   where survived = no: " << apriori_survived_no << endl;
    cout << "   where survived = yes: " << apriori_survived_yes << endl;

    // calculating likelihood for pclass
    double pclass_1_no = 0;
    double pclass_1_yes = 0;

    double pclass_2_no = 0;
    double pclass_2_yes = 0;

    double pclass_3_no = 0;
    double pclass_3_yes = 0;

    for (int i = 0; i < train_survived.size(); i++) {
        if (train_survived[i] == 0 && train_pclass[i] == 1) {
            pclass_1_no++;
        }
        else if (train_survived[i] == 1 && train_pclass[i] == 1) {
            pclass_1_yes++;
        }
        else if (train_survived[i] == 0 && train_pclass[i] == 2) {
            pclass_2_no++;
        }
        else if (train_survived[i] == 1 && train_pclass[i] == 2) {
            pclass_2_yes++;
        }
        else if (train_survived[i] == 0 && train_pclass[i] == 3) {
            pclass_3_no++;
        }
        else if (train_survived[i] == 1 && train_pclass[i] == 3) {
            pclass_3_yes++;
        }
    }

    pclass_1_no /= survived_no_count;
    pclass_1_yes /= survived_yes_count;
    pclass_2_no /= survived_no_count;
    pclass_2_yes /= survived_yes_count;
    pclass_3_no /= survived_no_count;
    pclass_3_yes /= survived_yes_count;

    cout << endl;
    cout << "Likelihood values for p(pclass|survived)" << endl;
    cout << "   where pclass = 1 and survived = no: " << pclass_1_no << endl;
    cout << "   where pclass = 1 and survived = yes: " << pclass_1_yes << endl;
    cout << "   where pclass = 2 and survived = no: " << pclass_2_no << endl;
    cout << "   where pclass = 2 and survived = yes: " << pclass_2_yes << endl;
    cout << "   where pclass = 3 and survived = no: " << pclass_3_no << endl;
    cout << "   where pclass = 3 and survived = yes: " << pclass_3_yes << endl;

    // calculating likelihood for sex
    double sex_0_no = 0;
    double sex_1_no = 0;

    double sex_0_yes = 0;
    double sex_1_yes = 0;

    for(int i = 0; i < train_survived.size(); i++) {
        if(train_survived[i] == 0 && train_sex[i] == 0) {
            sex_0_no++;
        }
        else if(train_survived[i] == 0 && train_sex[i] == 1) {
            sex_1_no++;
        }
        else if (train_survived[i] == 1 && train_sex[i] == 0) {
            sex_0_yes++;
        }
        else if (train_survived[i] == 1 && train_sex[i] == 1) {
            sex_1_yes++;
        }
    }

    sex_0_no /= survived_no_count;
    sex_1_no /= survived_no_count;
    sex_0_yes /= survived_yes_count;
    sex_1_yes /= survived_yes_count;

    cout << endl;
    cout << "Likelihood values for p(sex|survived)" << endl;
    cout << "   where sex = 0 and survived = no: " << sex_0_no << endl;
    cout << "   where sex = 0 and survived = yes: " << sex_0_yes << endl;
    cout << "   where sex = 1 and survived = no: " << sex_1_no << endl;
    cout << "   where sex = 1 and survived = yes: " << sex_1_yes << endl;

    // calculate likelihood for age
    double age_mean_no = 0;
    double age_mean_yes = 0;

    double age_var_no = 0;
    double age_var_yes = 0;

    for(int i = 0; i < train_survived.size(); i++) {
        if(train_survived[i] == 0) {
            age_mean_no += train_age[i];
        }
        else {
            age_mean_yes += train_age[i];
        }
    }

    age_mean_no /= survived_no_count;
    age_mean_yes /= survived_yes_count;

    for(int i = 0; i < train_survived.size(); i++) {
        if(train_survived[i] == 0) {
            age_var_no += pow((train_age[i] - age_mean_no), 2);
        }
        else {
            age_var_yes += pow((train_age[i] - age_mean_yes), 2);
        }
    }

    age_var_no /= survived_no_count - 1;
    age_var_yes /= survived_yes_count - 1;

    cout << endl;
    cout << "Likelihood values for p(age|survived)" << endl;
    cout << "   mean age for survived = no: " << age_mean_no << endl;
    cout << "   mean age for survived = yes: " << age_mean_yes << endl;
    cout << "   variance for survived = no: " << sqrt(age_var_no) << endl;
    cout << "   variance for survived = yes: " << sqrt(age_var_yes) << endl;
    auto end = chrono::steady_clock::now();

    cout << endl;
    cout << "Time taken to train the model with the training data: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " milliseconds." << endl;
    cout << endl;

    // testing model with the test data!
    cout << endl;
    cout << "Evaluating model with test data..." << endl;
    cout << endl;
    cout << setw(15) << "[0]" << "\t\t" << setw(15) << "[1]" << endl;

    for(int i = 0; i < test_survived.size() - 1; i++) {
        double num_s = 0;
        double num_p = 0;
        double denominator = 0;

        if (test_pclass[i] == 1 && test_sex[i] == 0) {
            num_s = pclass_1_yes * sex_0_yes * apriori_survived_yes * calc_age_lh(test_age[i], age_mean_yes, age_var_yes);
            num_p = pclass_1_no * sex_0_no * apriori_survived_no * calc_age_lh(test_age[i], age_mean_no, age_var_no);
            denominator = pclass_1_yes * sex_0_yes * apriori_survived_yes * calc_age_lh(test_age[i], age_mean_yes, age_var_yes)
                    + pclass_1_no * sex_0_no * apriori_survived_no * calc_age_lh(test_age[i], age_mean_no, age_var_no);
        }
        else if (test_pclass[i] == 1 && test_sex[i] == 1) {
            num_s = pclass_1_yes * sex_1_yes * apriori_survived_yes * calc_age_lh(test_age[i], age_mean_yes, age_var_yes);
            num_p = pclass_1_no * sex_1_no * apriori_survived_no * calc_age_lh(test_age[i], age_mean_no, age_var_no);
            denominator = pclass_1_yes * sex_1_yes * apriori_survived_yes * calc_age_lh(test_age[i], age_mean_yes, age_var_yes)
                          + pclass_1_no * sex_1_no * apriori_survived_no * calc_age_lh(test_age[i], age_mean_no, age_var_no);
        }
        else if (test_pclass[i] == 2 && test_sex[i] == 0) {
            num_s = pclass_2_yes * sex_0_yes * apriori_survived_yes * calc_age_lh(test_age[i], age_mean_yes, age_var_yes);
            num_p = pclass_2_no * sex_0_no * apriori_survived_no * calc_age_lh(test_age[i], age_mean_no, age_var_no);
            denominator = pclass_2_yes * sex_0_yes * apriori_survived_yes * calc_age_lh(test_age[i], age_mean_yes, age_var_yes)
                          + pclass_2_no * sex_0_no * apriori_survived_no * calc_age_lh(test_age[i], age_mean_no, age_var_no);
        }
        else if (test_pclass[i] == 2 && test_sex[i] == 1) {
            num_s = pclass_2_yes * sex_1_yes * apriori_survived_yes * calc_age_lh(test_age[i], age_mean_yes, age_var_yes);
            num_p = pclass_2_no * sex_1_no * apriori_survived_no * calc_age_lh(test_age[i], age_mean_no, age_var_no);
            denominator = pclass_2_yes * sex_1_yes * apriori_survived_yes * calc_age_lh(test_age[i], age_mean_yes, age_var_yes)
                          + pclass_2_no * sex_1_no * apriori_survived_no * calc_age_lh(test_age[i], age_mean_no, age_var_no);
        }
        else if (test_pclass[i] == 3 && test_sex[i] == 0) {
            num_s = pclass_3_yes * sex_0_yes * apriori_survived_yes * calc_age_lh(test_age[i], age_mean_yes, age_var_yes);
            num_p = pclass_3_no * sex_0_no * apriori_survived_no * calc_age_lh(test_age[i], age_mean_no, age_var_no);
            denominator = pclass_3_yes * sex_0_yes * apriori_survived_yes * calc_age_lh(test_age[i], age_mean_yes, age_var_yes)
                          + pclass_3_no * sex_0_no * apriori_survived_no * calc_age_lh(test_age[i], age_mean_no, age_var_no);
        }
        else if (test_pclass[i] == 3 && test_sex[i] == 1) {
            num_s = pclass_3_yes * sex_1_yes * apriori_survived_yes * calc_age_lh(test_age[i], age_mean_yes, age_var_yes);
            num_p = pclass_3_no * sex_1_no * apriori_survived_no * calc_age_lh(test_age[i], age_mean_no, age_var_no);
            denominator = pclass_3_yes * sex_1_yes * apriori_survived_yes * calc_age_lh(test_age[i], age_mean_yes, age_var_yes)
                          + pclass_3_no * sex_1_no * apriori_survived_no * calc_age_lh(test_age[i], age_mean_no, age_var_no);
        }

        double probability_survived = num_s / denominator;
        double probability_perished = num_p / denominator;

        cout << "[" << right << setw(5) << i << "]" << right << setw(10) << probability_perished << "\t\t" << right << setw(10) <<  probability_survived << endl;
    }
    return 0;
}
