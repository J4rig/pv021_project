#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <assert.h>

const double LEARNING_RATE = 0.01;

using namespace std;

class Matrix {
public:
	int rows;
	int columns;
	vector<double> data;

	Matrix() : rows(0), columns(0) {}

	Matrix(const int rows, const int columns) : rows(rows), columns(columns) {
		data.reserve(rows * columns);
		for (int i = 0; i < rows * columns; i++) {
			data.emplace_back(0.0);
		}
	}

	void randomize() {
		for (int i = 0; i < rows * columns; i++) {
			data[i] = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX) - 0.5) * 0.1; //TODO
		}
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

	void print() {
		for (int i = 0; i < rows * columns; i++) {
			if (i % columns == 0) {
				cout << '\n';
			}
			cout << data[i]  << ' ';
		}
		cout << '\n';
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

		weights.randomize();
		biases.randomize();
	};		
};

class Network {
public:
	vector<Layer> layers;

	Network(const int input_cnt, const int hiden_cnt, const int out_cnt, const int batch_size) {
		layers.emplace_back(hiden_cnt, input_cnt);
		layers.emplace_back(out_cnt, hiden_cnt);

		hiden_inner_p = Matrix(batch_size, hiden_cnt);
		hiden_out = Matrix(batch_size, hiden_cnt);

		hiden_derivate_out = Matrix(batch_size, hiden_cnt);
		hiden_gradients = Matrix(input_cnt, hiden_cnt);
		hiden_gradient_part = Matrix(batch_size, hiden_cnt);
		hiden_gradients_bias = Matrix(1, hiden_cnt);

		out_inner_p = Matrix(batch_size, out_cnt);
		out = Matrix(batch_size, out_cnt);

		out_gradients = Matrix(hiden_cnt, out_cnt);
		out_gradient_part = Matrix(batch_size, out_cnt);
		out_gradients_bias = Matrix(1, out_cnt);
	}

	void forwardPropagate(const Matrix &input) {
		assert(layers.size() == 2);

		hiden_inner_p.multiply(input, layers[0].weights);
		hiden_inner_p.addRow(layers[0].biases);
		ReLU(hiden_inner_p,hiden_out);

		out_inner_p.multiply(hiden_out, layers[1].weights);
		out_inner_p.addRow(layers[1].biases);
		Softmax(out_inner_p, out);
	}

	void backPropagate(const Matrix& input, const Matrix& desired_out) {
		assert(desired_out.columns == out.columns);
		assert(desired_out.rows == out.rows);

		out_gradient_part.subtract(out,desired_out);
		out_gradients.multiplyTransposed(hiden_out, out_gradient_part);

		hiden_gradient_part.multiplyWithTransposed(out_gradient_part, layers[1].weights);

		ReLUDerived(hiden_inner_p, hiden_derivate_out);
		hiden_gradient_part.elementMultiply(hiden_derivate_out);
		hiden_gradients.multiplyTransposed(input, hiden_gradient_part);		

		for (int i = 0; i < layers[0].weights.rows * layers[0].weights.columns; i++) {
			layers[0].weights.data[i] -= hiden_gradients.data[i] * LEARNING_RATE;
		}

		for (int i = 0; i < layers[1].weights.rows * layers[1].weights.columns; i++) {
			layers[1].weights.data[i] -= out_gradients.data[i] * LEARNING_RATE;
		}

		for (int j = 0; j < layers[1].biases.columns; j++) {
			double grad = 0.0;
			for (int i = 0; i < out_gradient_part.rows; i++)
				grad += out_gradient_part.data[i * out_gradient_part.columns + j];
			grad /= out_gradient_part.rows;
			layers[1].biases.data[j] -= LEARNING_RATE * grad;
		}

		for (int j = 0; j < layers[0].biases.columns; j++) {
			double grad = 0.0;
			for (int i = 0; i < hiden_gradient_part.rows; i++)
				grad += hiden_gradient_part.data[i * hiden_gradient_part.columns + j];
			grad /= hiden_gradient_part.rows;
			layers[0].biases.data[j] -= LEARNING_RATE * grad;
		}
	}

	//Cross entropy
	void errorFunction(const Matrix& expected_values) {
		double result = 0.0;
		double eps = 1e-12;
		for (int i = 0; i < expected_values.rows * expected_values.columns; i++) {
			result -= expected_values.data[i] * log(out.data[i] + eps);
		}
		cout << result / expected_values.rows << '\n';
	}

	void printOut() {
		out.print();
	}

private:
	Matrix hiden_inner_p;
	Matrix hiden_out;
	Matrix hiden_derivate_out;
	Matrix hiden_gradients;
	Matrix hiden_gradient_part;
	Matrix hiden_gradients_bias;

	Matrix out_inner_p;
	Matrix out;
	Matrix out_gradients;
	Matrix out_gradient_part;
	Matrix out_gradients_bias;
	//vector<Matrix>  
};

int main() {

	int batch_size = 5;

	Network network = Network(2,2,4, batch_size);

	for (int l = 0; l < network.layers.size(); l++) {
		network.layers[l].weights.print();
		cout << "\n";
	}

	Matrix in = Matrix(batch_size, 2);

	in.data = { 0, 0,
			    0, 1,
			    1, 0,
			    1, 1,
			    0, 1 };

	Matrix out = Matrix(batch_size, 4);

	out.data = { 0, 0, 0, 1,
			     0, 0, 1, 0,
			     0, 1, 0, 0,
			     1, 0, 0, 0,
				 0, 0, 1, 0 };

	int n = 10000;

	for (int i = 0; i < n; i++) {
		if (i == 0) {
			cout << "Starting values \n";
		}
		else if (i == n - 1) {
			cout << "Values after " << n << " training loops\n";
		}
		network.forwardPropagate(in);
		if (i == 0 || i == n - 1) {
			network.printOut();
		}
		network.backPropagate(in, out);
		//network.errorFunction(out);
	}
	
	return 0;
}