// name: aditi chaudhari
// course: cs 4375.003
// assigment name: HW1 (data_exploration)

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

void print_stats(vector<double> v);
double sum(vector<double> v);
double mean(vector<double> v);
double median(vector<double> v);
vector<double> range(vector<double> v);
double cov(vector<double> v1, vector<double> v2);
double cor(vector<double> v1, vector<double> v2);

int main(int argc, char** argv) {

    ifstream inFS;  // Input file stream
    string line;
    string rm_in, medv_in;
    const int MAX_LEN = 1000;
    vector<double> rm(MAX_LEN);
    vector<double> medv(MAX_LEN);

    // Try to open file
    cout << "Opening file Boston.csv" << endl;

    inFS.open("Boston.csv");
    if(!inFS.is_open()) {
        cout << "Could not open file Boston.csv." << endl;
        return 1; // 1 indicates error
    }

    // Can now use inFS stream like cin stream
    // Boston.csv should contain two doubles

    cout << "Reading line 1" << endl;
    getline(inFS, line);

    // echo heading
    cout << "heading: " << line << endl;

    int numObservations = 0;

    while (inFS.good()) {
        getline(inFS, rm_in, ',');
        getline(inFS, medv_in, '\n');

        rm.at(numObservations) = stof(rm_in);
        medv.at(numObservations) = stof(medv_in);

        numObservations++;
    }

    rm.resize(numObservations);
    medv.resize(numObservations);

    cout << "new length " << rm.size() << endl;

    cout << "Closing file Boston.csv." << endl;
    inFS.close();

    cout << "Number of records: " << numObservations << endl;

    cout << "\nStats for rm" << endl;
    print_stats(rm);

    cout << "\nStats for medv" << endl;
    print_stats(medv);

    cout << "\nCovariance = " << cov(rm, medv) << endl;

    cout << "\nCorrelation = " << cor(rm, medv) << endl;

    cout << "\nProgram terminated.";

    return 0;
}

void print_stats(vector<double> v) {
    vector<double> rangev = range(v);

    cout << "Sum: " << sum(v) << endl;
    cout << "Mean: " << mean(v) << endl;
    cout << "Median: " << median(v) << endl;
    cout << "Range: " << "[" << rangev.at(0) << ", " << rangev.at(1) << "]" << endl;
}

double sum(vector<double> v) {
    double sum = 0;
    for(int i = 0; i < v.size(); i++) {
        sum += v.at(i);
    }
    return sum;
}

double mean(vector<double> v) {
    double average = 0;
    double vector_sum = sum(v);
    average = vector_sum / v.size();
    return average;
}

double median(vector<double> v) {
    int size = v.size();
    int n = size / 2;
    nth_element(v.begin(), v.begin() + n, v.end());
    return v.at(n);
}

vector<double> range(vector<double> v) {
    vector<double> rangev;

    double max = INT32_MIN;
    double min = INT32_MAX;

    for(int i = 0; i < v.size(); i++) {
        if (v.at(i) < min) {
            min = v.at(i);
        }
        if (v.at(i) > max) {
            max = v.at(i);
        }
    }

    rangev.push_back(min);
    rangev.push_back(max);
    return rangev;
}

double cov(vector<double> v1, vector<double> v2) {
    double x_bar = mean(v1);
    double y_bar = mean(v2);

    // holds Xi - Xbar
    double x = 0;

    // holds Yi - Ybar
    double y = 0;

    // holds sigma((Xi - Xbar)(Yi - Ybar))
    double sigma_x_y = 0;

    for(int i = 0; i < v1.size(); i++) {
        x = v1.at(i) - x_bar;
        y = v2.at(i) - y_bar;

        sigma_x_y += x * y;
    }

    double covariance = sigma_x_y / ((double)v1.size() - 1);
    return covariance;
}

double cor(vector<double> v1, vector<double> v2) {
    double x_bar = mean(v1);
    double y_bar = mean(v2);

    // holds Xi - Xbar
    double x = 0;

    // holds Yi - Ybar
    double y = 0;

    // sigma((Xi - Xbar)(Yi - Ybar))
    double numerator = 0;

    // sigma(Xi - Xbar)^2
    double sigma_x = 0;

    // sigma(Yi - Ybar)^2
    double sigma_y = 0;

    for(int i = 0; i < v1.size(); i++) {
        x = v1.at(i) - x_bar;
        y = v2.at(i) - y_bar;

        numerator += x * y;

        sigma_x += pow((v1.at(i) - x_bar), 2);
        sigma_y += pow((v2.at(i) - y_bar),2);
    }

    double denominator = sqrt(sigma_x * sigma_y);
    double correlation = numerator / denominator;
    return correlation;
}


