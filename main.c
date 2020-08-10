#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mnist_c_funcs/mnist_funcs.h"
#include "mnist_c_funcs/yann2c/deep_network.h"
#include "mnist_c_funcs/yann2c/matrix.h"

int main (void)
{
       	struct matrix_array *input =
				matrix_alloc_mnist_images("train-images.idx3-ubyte", 10); 

       	if(input == NULL) return 1;

       	struct matrix_array *expected_output =
				matrix_alloc_mnist_labels("train-labels.idx1-ubyte", 10);

	if(expected_output == NULL) { matrix_array_free(input); return 1; }

	struct nn_array *nn = nn_create();
	if(nn == NULL)
	{
		matrix_array_free(input);
		matrix_array_free(expected_output);
		return 1;
	}
		//		size	input	batch	actv func	dropout_rate
	nn_add_layer(nn, 	40, 	784, 	10, 	ReLU, 		0.5);
	nn_add_layer(nn, 	10, 	0, 	10, 	NULL, 		0.0);

	matrix_fill_rng(nn->head->weights, 0.1, 0.9);
	matrix_fill_rng(nn->tail->weights, 0.1, 0.9);

	struct matrix_node* ptr_in = input->head;
	struct matrix_node* ptr_out = expected_output->head;

	for(int i = 0; i < 1000; ++i)
	{
	       nn_backpropagation(nn, ptr_in->matrix, ptr_out->matrix, 0.01, 1, 1); 
	       ptr_in = ptr_in->next;
	       ptr_out = ptr_out->next;
	       if(ptr_in == NULL || ptr_out == NULL) break;
	}
	
	matrix_array_free(input);
	matrix_array_free(expected_output);
	nn_free(nn);
	return 0;
}
