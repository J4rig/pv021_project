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

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < m1.columns; j++) {
				double x = m1.data[i * m1.columns + j];
				for (int k = 0; k < columns; k++) {
					data[i * columns + k] += x * m2.data[j * columns + k];
				}
			}
		}
	}

	//m1 is be used as a transposed matrix, m2 is normal
	void multiplyTransposed(const Matrix& m1, const Matrix& m2) {
		assert(m1.rows == m2.rows);
		assert(m1.columns == rows);
		assert(m2.columns == columns);

		fill(data.begin(), data.end(), 0.0);

		for (int i = 0; i < m1.rows; i++) {
			for (int j = 0; j < rows; j++) {
				double x = m1.data[i * rows + j];
				for (int k = 0; k < columns; k++) {
					data[j * columns + k] += x * m2.data[i * columns + k];
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

	void elementMultiply(const Matrix& m, Matrix& m_out) const {
		assert(rows == m.rows);
		assert(rows == m_out.rows);
		assert(columns == m.columns);
		assert(columns == m_out.columns);

		fill(m_out.data.begin(), m_out.data.end(), 0.0);

		for (int i = 0; i < rows * columns; i++) {
			m_out.data[i] = data[i] * m.data[i];
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
				cout << "\n";
			}
			cout << data[i]  << ' ';
		}
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
	double max = *max_element(m.data.begin(), m.data.end());

	double out_sum = 0.0;
	for (int i = 0; i < m.rows * m.columns; i++) {
		out_sum += exp(m.data[i] - max);
	}

	for (int i = 0; i < m.rows * m.columns; i++) {
		m_out.data[i] = exp(m.data[i] - max) / out_sum;
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

		hiden_derivate_out = Matrix(1, hiden_cnt);
		hiden_gradients = Matrix(input_cnt, hiden_cnt);
		hiden_gradient_part = Matrix(1, hiden_cnt);

		out_inner_p = Matrix(batch_size, out_cnt);
		out = Matrix(batch_size, out_cnt);

		out_gradients = Matrix(hiden_cnt, out_cnt);
		out_gradient_part = Matrix(1, out_cnt);
	}

	//void forwardPropagate(const Matrix &input) {
	//	assert(layers.size() == 2);

	//	input.multiply(layers[0].weights, hiden_inner_p);
	//	hiden_inner_p.addRow(layers[0].biases);
	//	ReLU(hiden_inner_p,hiden_out);

	//	hiden_out.multiply(layers[1].weights, out_inner_p);
	//	out_inner_p.addRow(layers[1].biases);
	//	Softmax(out_inner_p, out);

	//	//out.print();
	//}

	//void backPropagate(const Matrix& input, const Matrix& desired_out) {
	//	assert(desired_out.columns == out.columns);
	//	assert(desired_out.rows == out.rows);

	//	out_gradient_part.subtract(out,desired_out);
	//	out_gradients.multiplyTransposed(hiden_out, out_gradient_part);

	//	out_gradient_part.multiplyTransposed(layers[1].weights, hiden_gradient_part);
	//	ReLUDerived(hiden_inner_p, hiden_derivate_out);
	//	hiden_gradient_part.elementMultiply(hiden_derivate_out, hiden_gradient_part);
	//	input.sumOuterProducts(hiden_gradient_part, hiden_gradients);

	//	for (int i = 0; i < layers[0].weights.rows * layers[0].weights.columns; i++) {
	//		layers[0].weights.data[i] -= hiden_gradients.data[i] * LEARNING_RATE;
	//	}

	//	for (int i = 0; i < layers[1].weights.rows * layers[1].weights.columns; i++) {
	//		layers[1].weights.data[i] -= out_gradients.data[i] * LEARNING_RATE;
	//	}
	//}

	//Cross entropy
	void errorFunction(const Matrix& expected_values) {
		double result = 0.0;
		double eps = 1e-12;
		for (int i = 0; i < expected_values.rows * expected_values.columns; i++) {
			result -= expected_values.data[i] * log(out.data[i] + eps);
		}
		cout << result << '\n';
	}

private:
	Matrix hiden_inner_p;
	Matrix hiden_out;
	Matrix hiden_derivate_out;
	Matrix hiden_gradients;
	Matrix hiden_gradient_part;

	Matrix out_inner_p;
	Matrix out;
	Matrix out_gradients;
	Matrix out_gradient_part;
	//vector<Matrix>  
};

int main() { // chyba bias

	/*Network network = Network(2,2,4);

	for (int l = 0; l < network.layers.size(); l++) {
		network.layers[l].weights.print();
		cout << "\n";
	}

	Matrix in = Matrix(1, 2);
	Matrix out = Matrix(1, 4);
	in.randomize();
	out.data = { 1.0,0.0,0.0,0.0 };

	for (int i = 0; i < 10; i++) {
		network.forwardPropagate(in);
		network.backPropagate(in, out);
		network.errorFunction(out);
	}*/

	Matrix x = Matrix(5, 4);
	Matrix y = Matrix(5, 3);
	Matrix w = Matrix(3, 5);

	Matrix z = Matrix(3, 4);

	x.data = { 1, 2, 3, 4,
			   5, 6, 7, 8,
			   9, 10, 11, 12,
			   13, 14, 15, 16,
			   17, 18, 19, 20 };

	y.data = { 1, 2, 3,
			   4, 5, 6,
			   7, 8, 9,
			   10, 11, 12,
			   13, 14, 15 };

	w.data = y.data;

	z.multiplyTransposed(y, x);

	x.print();
	cout << '\n';

	y.print();
	cout << '\n';

	z.print();
	cout << '\n';

	z.multiply(w, x);

	z.print();
	
	return 0;
}