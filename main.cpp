#include <iostream>
#include <fstream>
#include <string>

using namespace std;



int main() {
	cout << "hello world\n";

	ifstream training_vectors_file("fashion_mnist_train_vectors.csv");

	ifstream training_labels_file("fashion_mnist_train_labels.csv");

	string vector;

	string label;

	getline(training_vectors_file, vector);
	cout << vector << "\n";	

	getline(training_labels_file, label);
	cout << label << "\n";
}