#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mnist_c_funcs/mnist_funcs.h"
#include "mnist_c_funcs/yann2c/deep_network.h"
#include "mnist_c_funcs/yann2c/matrix.h"

int main (void)
{
       	matrix_t *input = matrix_alloc_mnist_images("train-images.idx3-ubyte"); 
       	if(input == NULL) return 1;

       	matrix_t *expected_output = matrix_alloc_mnist_labels("train-labels.idx1-ubyte");
	if(expected_output == NULL) { matrix_free(input); return 1; }

	struct nn_array *nn = nn_create();
	if(nn == NULL)
	{
		matrix_free(input);
		matrix_free(expected_output);
		return 1;
	}
	nn_add_layer(nn, 40, 784, 60000, ReLU, 0.5);
	nn_add_layer(nn, 10, 0, 0, NULL, 0.0);

	nn_backpropagation(nn, input, expected_output, 0.01, 1, 1);

	nn_free(nn);
	matrix_free(input);
	matrix_free(expected_output);
	return 0;
}
