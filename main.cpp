#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <random>
#include <fstream>
#include <string>

using namespace std;

double randomizeNormal(const double mean, const double deviation) {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::normal_distribution<double> dist(mean, deviation);
	return dist(gen);
}

class Matrix {
public:
	int rows;
	int columns;
	vector<double> data;

	Matrix() : rows(0), columns(0) {}

	Matrix(const int rows, const int columns) : rows(rows), columns(columns) {
		data.resize(rows * columns, 0.0);
	}

	void multiply(const Matrix &m1, const Matrix &m2) {
		assert(m1.columns == m2.rows);
		assert(m1.rows == rows);
		assert(m2.columns == columns);

		fill(data.begin(), data.end(), 0.0);

		int X = rows;
		int Y = m1.columns;
		int Z = columns;

		for (int i = 0; i < X; i++) {
			for (int j = 0; j < Y; j++) {
				double x = m1.data[i * Y + j];
				for (int k = 0; k < Z; k++) {
					data[i * Z + k] += x * m2.data[j * Z + k];
				}
			}
		}
	}

	//m1 is used as a transposed matrix, m2 is normal
	void multiplyTransposed(const Matrix& m1, const Matrix& m2) {
		assert(m1.rows == m2.rows);
		assert(m1.columns == rows);
		assert(m2.columns == columns);

		fill(data.begin(), data.end(), 0.0);

		int X = m1.rows;
		int Y = rows;
		int Z = columns;

		for (int i = 0; i < X; i++) {
			for (int j = 0; j < Y; j++) {
				double x = m1.data[i * Y + j];
				for (int k = 0; k < Z; k++) {
					data[j * Z + k] += x * m2.data[i * Z + k];
				}
			}
		}
	}

	//m2 is be used as a transposed matrix, m1 is normal
	void multiplyWithTransposed(const Matrix& m1, const Matrix& m2) {
		assert(m1.columns == m2.columns);
		assert(m1.rows == rows);
		assert(m2.rows == columns);

		fill(data.begin(), data.end(), 0.0);

		int X = rows;
		int Y = columns;
		int Z = m1.columns;

		for (int i = 0; i < X; i++) {
			for (int j = 0; j < Y; j++) {
				for (int k = 0; k < Z; k++) {
					data[i * Y + j] += m1.data[i * Z + k] * m2.data[j * Z + k];
				}
			}
		}
	}

	void subtract(const Matrix& m1, const Matrix& m2) {	
		assert(m1.rows == m2.rows);
		assert(m1.columns == m2.columns);
		assert(m1.rows == rows);
		assert(m1.columns == columns);

		for (int i = 0; i < rows * columns; i++) {
			data[i] = m1.data[i] - m2.data[i];
		}
	}

	void elementMultiply(const Matrix& m)  {
		assert(rows == m.rows);
		assert(columns == m.columns);

		for (int i = 0; i < rows * columns; i++) {
			data[i] *= m.data[i];
		}
	}

	void addRow(const Matrix& row) {
		assert(columns == row.columns);
		assert(row.rows == 1);

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				data[i * columns + j] += row.data[j];
			}
		}
	}

	void sumRows(const Matrix& m) {
		assert(rows == 1);
		assert(columns == m.columns);

		fill(data.begin(), data.end(), 0.0);

		for (int j = 0; j < m.columns; j++) {
			for (int i = 0; i < m.rows; i++) {
				data[j] += m.data[i * m.columns + j];
			}
		}
	}

	void print() {
		for (int i = 0; i < rows * columns; i++) {
			if (i % columns == 0) {
				std::cout << '\n';
			}
			std::cout << data[i]  << ' ';
		}
		std::cout << '\n';
	}

};

void ReLU(const Matrix& m, Matrix& m_out) {
	const int n = m.rows * m.columns;
	for (int i = 0; i < n; i++) {
		m_out.data[i] = std::max(0.0, m.data[i]);
	}
}

void ReLUDerived(const Matrix& m, Matrix& m_out) {
	for (int i = 0; i < m.rows * m.columns; i++) {
		m_out.data[i] = (m.data[i] > 0) ? 1.0 : 0.0;
	}
}

void Softmax(const Matrix& m, Matrix& m_out) {
	assert(m.rows == m_out.rows);
	assert(m.columns == m_out.columns);

	for (int i = 0; i < m.rows; i++) {

		double max = *max_element(m.data.begin() + i * m.columns, m.data.begin() + (i + 1) * m.columns);

		double out_sum = 0.0;
		for (int j = 0; j < m.columns; j++) {
			out_sum += exp(m.data[i * m.columns + j] - max);
		}

		for (int j = 0; j < m.columns; j++) {
			m_out.data[i * m.columns + j] = exp(m.data[i * m.columns + j] - max) / out_sum;
		}
	}
}

class Layer {
public:
	int neurons;
	int inputs;

	Matrix weights;
	Matrix biases;

	Layer(int neurons, int inputs) :
		neurons(neurons),
		inputs(inputs),
		weights(inputs, neurons),
		biases(1, neurons) {
	};

	void glorotWeightInit() {
		for (double &weight : weights.data) {
			weight = randomizeNormal(0.0, 2.0 / (neurons + inputs));
		}
	}

	void heWeightInit() {
		for (double &weight : weights.data) {
			weight = randomizeNormal(0.0, 2.0 / neurons);
		}
	}

};

class Network {
public:
	vector<Layer> layers;

	Network(const int input_cnt, const int hidden_cnt, const int out_cnt, const int batch_size) {
		layers.emplace_back(hidden_cnt, input_cnt);
		layers.emplace_back(out_cnt, hidden_cnt);

		layers[0].heWeightInit();
		layers[1].glorotWeightInit();

		hidden_inner_p = Matrix(batch_size, hidden_cnt);
		hidden_out = Matrix(batch_size, hidden_cnt);

		hidden_derivate_out = Matrix(batch_size, hidden_cnt);
		hidden_gradients = Matrix(input_cnt, hidden_cnt);
		hidden_gradient_part = Matrix(batch_size, hidden_cnt);
		hidden_gradients_bias = Matrix(1, hidden_cnt);

		hidden_delta_prev = Matrix(input_cnt, hidden_cnt);
		hidden_delta_bias_prev = Matrix(1, hidden_cnt);

		out_inner_p = Matrix(batch_size, out_cnt);
		out = Matrix(batch_size, out_cnt);

		out_gradients = Matrix(hidden_cnt, out_cnt);
		out_gradient_part = Matrix(batch_size, out_cnt);
		out_gradients_bias = Matrix(1, out_cnt);

		out_delta_prev = Matrix(hidden_cnt, out_cnt);
		out_delta_bias_prev = Matrix(1, out_cnt);

	}

	void forwardPropagate(const Matrix &input) {
		assert(layers.size() == 2);

		hidden_inner_p.multiply(input, layers[0].weights);
		hidden_inner_p.addRow(layers[0].biases);
		ReLU(hidden_inner_p,hidden_out);

		out_inner_p.multiply(hidden_out, layers[1].weights);
		out_inner_p.addRow(layers[1].biases);
		Softmax(out_inner_p, out);
	}

	void backPropagate(const Matrix& input, const Matrix& desired_out, double LEARNING_RATE, double momentum_influence) {
		assert(desired_out.columns == out.columns);
		assert(desired_out.rows == out.rows);

		out_gradient_part.subtract(out,desired_out);
		out_gradients.multiplyTransposed(hidden_out, out_gradient_part);

		out_gradients_bias.sumRows(out_gradient_part);

		hidden_gradient_part.multiplyWithTransposed(out_gradient_part, layers[1].weights);

		ReLUDerived(hidden_inner_p, hidden_derivate_out);
		hidden_gradient_part.elementMultiply(hidden_derivate_out);
		hidden_gradients.multiplyTransposed(input, hidden_gradient_part);		

		hidden_gradients_bias.sumRows(hidden_gradient_part);


		for (int i = 0; i < layers[1].weights.rows * layers[1].weights.columns; i++) {
			double scaled_delta = out_gradients.data[i];
			double delta = -LEARNING_RATE * scaled_delta + out_delta_prev.data[i] * momentum_influence;
			layers[1].weights.data[i] += delta;
			out_delta_prev.data[i] = delta;
		}

		for (int i = 0; i < layers[1].biases.columns; i++) {
			double scaled_delta_bias = out_gradients_bias.data[i];
			double delta = -LEARNING_RATE * scaled_delta_bias + out_delta_bias_prev.data[i] * momentum_influence;
			layers[1].biases.data[i] += delta;
			out_delta_bias_prev.data[i] = delta;
		}

		for (int i = 0; i < layers[0].weights.rows * layers[0].weights.columns; i++) {
			double scaled_delta = hidden_gradients.data[i];
			double delta = -LEARNING_RATE * scaled_delta + hidden_delta_prev.data[i] * momentum_influence;
			layers[0].weights.data[i] += delta;
			hidden_delta_prev.data[i] = delta;
		}

		for (int i = 0; i < layers[0].biases.columns; i++) {
			double scaled_delta_bias = hidden_gradients_bias.data[i];
			double delta = -LEARNING_RATE * scaled_delta_bias + hidden_delta_bias_prev.data[i] * momentum_influence;
			layers[0].biases.data[i] += delta;
			hidden_delta_bias_prev.data[i] = delta;
		}
	}

	//Cross entropy
	void errorFunction(const Matrix& expected_values) {
		double result = 0.0;
		double eps = 1e-12;
		for (int i = 0; i < expected_values.rows * expected_values.columns; i++) {
			result -= expected_values.data[i] * log(out.data[i] + eps);
		}
		std::cout << result / (expected_values.rows * expected_values.columns)<< '\n';
	}

	void printOut() {
		out.print();
		std::cout << '\n';
	}

	int compareOutput(const Matrix &expected_values) {
		int c = 0;
		for (int i = 0; i < out.rows; i++) {

			auto out_begin = out.data.begin() + i * out.columns;
			auto out_end = out_begin + out.columns;

			auto exp_begin = expected_values.data.begin() + i * out.columns;
			auto exp_end = exp_begin + out.columns;


			int pred_idx = std::distance(out_begin, std::max_element(out_begin, out_end));
			int true_idx = std::distance(exp_begin, std::max_element(exp_begin, exp_end));

			if (pred_idx == true_idx)
				c++;
		}
		return c;
	}

private:
	Matrix hidden_inner_p;
	Matrix hidden_out;
	Matrix hidden_derivate_out;
	Matrix hidden_gradients;
	
	Matrix hidden_gradient_part;
	Matrix hidden_gradients_bias;

	Matrix hidden_delta_prev;
	Matrix hidden_delta_bias_prev;

	Matrix out_inner_p;
	Matrix out;
	Matrix out_gradients;
	Matrix out_gradient_part;
	Matrix out_gradients_bias;

	Matrix out_delta_prev;
	Matrix out_delta_bias_prev;
};


void normalizeData(const string file_name, Matrix &m) {
	std::cout << "Normalization starts\n";

	ifstream training_vectors_file(file_name);

	string input_line;
	int c = 0;

	while (getline(training_vectors_file, input_line)) {
		size_t prev = 0;
		size_t pos;

		while ((pos = input_line.find_first_of(",", prev)) != std::string::npos) {
			if (pos > prev) m.data[c] = stod(input_line.substr(prev, pos - prev)) / 255;
			prev = pos + 1;
			c++;
		}
		if (prev < input_line.length()) {
			m.data[c] = stod(input_line.substr(prev, std::string::npos)) / 255;
			c++;
		}
	}
	assert(c == m.rows * m.columns);
	std::cout << "Normalized " << c << " inputs from " << file_name << "\n";
}

void loadLabels(const string file_name, const vector<vector<double>> &outputs, Matrix &labels ) {
	std::cout << "Loading labels\n";
	int c = 0;

	ifstream train_labels(file_name);
	string label_line;

	for (c = 0; getline(train_labels, label_line); c++) {
		int i = stol(label_line);
		copy(outputs[i].begin(), outputs[i].end(), labels.data.begin() + c * 10);
	}
	std::cout << "Loaded " << c << " labels\n";
}

int main() {
	int input_size = 784;
	int output_size = 10;
	int train_size = 60000;
	int test_size = 10000;

	int hidden_layer_size = 70;

	int batch_size = 10;
	
	Matrix batch = Matrix(batch_size, input_size);

	Network network = Network(input_size, hidden_layer_size, output_size, batch_size);

	vector<int> batch_results(batch_size); // outputs produced by batch;

	Matrix desired_outputs(batch_size, output_size);

	vector<vector<double>> outputs {{ 1,0,0,0,0,0,0,0,0,0 },
									{ 0,1,0,0,0,0,0,0,0,0 },
									{ 0,0,1,0,0,0,0,0,0,0 },
									{ 0,0,0,1,0,0,0,0,0,0 },
									{ 0,0,0,0,1,0,0,0,0,0 },
									{ 0,0,0,0,0,1,0,0,0,0 },
									{ 0,0,0,0,0,0,1,0,0,0 },
									{ 0,0,0,0,0,0,0,1,0,0 },
									{ 0,0,0,0,0,0,0,0,1,0 },
									{ 0,0,0,0,0,0,0,0,0,1 }};

	

	Matrix normalized_train_data(train_size, input_size);
	Matrix normalized_test_data(test_size, input_size);

	Matrix train_labels(train_size, output_size);
	Matrix test_labels(test_size, output_size);

	normalizeData("fashion_mnist_train_vectors.csv", normalized_train_data);
	normalizeData("fashion_mnist_test_vectors.csv", normalized_test_data);

	loadLabels("fashion_mnist_train_labels.csv", outputs, train_labels);
	loadLabels("fashion_mnist_test_labels.csv", outputs, test_labels);


	std::cout << "Training starts\n";

	int batch_start;
	int correct;

	double LEARNING_RATE = 0.008;

	for (int i = 0; i < 5; i++) {
		correct = 0;
											//60000
		for (batch_start = 0; batch_start < train_size; batch_start += batch_size) {
			copy(normalized_train_data.data.begin() + input_size * batch_start,
				normalized_train_data.data.begin() + input_size * (batch_start + batch_size),
				batch.data.begin());

			copy(train_labels.data.begin() + output_size * batch_start,
				train_labels.data.begin() + output_size * (batch_start + batch_size),
				desired_outputs.data.begin());

			network.forwardPropagate(batch);

			correct += network.compareOutput(desired_outputs);

			network.backPropagate(batch, desired_outputs, LEARNING_RATE / (1 + (batch_start) / 10000), 0.4);
		}

		std::cout << "Training " << i + 1 << " done, correct comparisons : " << correct << "\n";
	}
	std::cout << "Testing starts\n";

	Matrix testing_input = Matrix(batch_size, input_size);
	Matrix testing_output = Matrix(batch_size, output_size);

	correct = 0;
										//10000
	for (batch_start = 0; batch_start < test_size; batch_start += batch_size) {
		copy(normalized_test_data.data.begin() + input_size * batch_start,
			 normalized_test_data.data.begin() + input_size * (batch_start + batch_size),
			 testing_input.data.begin());

		copy(test_labels.data.begin() + output_size * batch_start,
			 test_labels.data.begin() + output_size * (batch_start + batch_size),
			 testing_output.data.begin());

		network.forwardPropagate(testing_input);

		correct += network.compareOutput(testing_output);
	}
	std::cout << "Testing done, accuracy: " << static_cast<double>(correct)/test_size << "\n";
	return 0;
}
