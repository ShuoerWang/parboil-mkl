/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <string.h>

#include "file.h"
#include "convert_dataset.h"



int main(int argc, char** argv) {
	struct pb_TimerSet timers;
	struct pb_Parameters *parameters;
	
	
	
	
	
	printf("CPU-based sparse matrix vector multiplication****\n");
	printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and Shengzhao Wu<wu14@illinois.edu>\n");
	printf("This version maintained by Chris Rodrigues  ***********\n");
	parameters = pb_ReadParameters(&argc, argv);
	if ((parameters->inpFiles[0] == NULL) || (parameters->inpFiles[1] == NULL))
    {
      fprintf(stderr, "Expecting two input filenames\n");
      exit(-1);
    }
	
	pb_InitializeTimerSet(&timers);
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	//parameters declaration
	int len;
	int depth;
	int dim;
	int pad=1;
	int nzcnt_len;
	
	//host memory allocation
	//matrix
	float *h_data;
	int *h_indices;
	int *h_ptr;
	int *h_perm;
	int *h_nzcnt;
	//vector
	float *h_Ax_vector;
    float *h_x_vector;
	
	
    //load matrix from files
	pb_SwitchToTimer(&timers, pb_TimerID_IO);
	//inputData(parameters->inpFiles[0], &len, &depth, &dim,&nzcnt_len,&pad,
	//    &h_data, &h_indices, &h_ptr,
	//    &h_perm, &h_nzcnt);
	int col_count;
	coo_to_jds(
		parameters->inpFiles[0], // bcsstk32.mtx, fidapm05.mtx, jgl009.mtx
		1, // row padding
		pad, // warp size
		1, // pack size
		1, // is mirrored?
		0, // binary matrix
		1, // debug level [0:2]
		&h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
		&col_count, &dim, &len, &nzcnt_len, &depth
	);		

  h_Ax_vector=(float*)malloc(sizeof(float)*dim);
  h_x_vector=(float*)malloc(sizeof(float)*dim);
  input_vec( parameters->inpFiles[1], h_x_vector,dim);	

  sparse_matrix_t A;
  int *rowstr;
  int *colidx;
  float **mat;
  float *values;
  
  int mkl = 1;
  if (mkl) {
	rowstr = (int*)malloc(sizeof(int)*(dim+1));
	int nz = 0;
	for (int i = 0; i < dim; i++) {
		nz += h_nzcnt[i];
	}
	colidx = (int*)malloc(sizeof(int)*nz);
	values = (float*)malloc(sizeof(float)*nz);

	mat = (float**)malloc(sizeof(float*)*dim);
	for (int i = 0; i < dim; i++) {
		mat[i] = (float*)malloc(sizeof(float)*dim);
		memset(mat[i], 0, sizeof(float)*dim);
	}
	for (int i = 0; i < dim; i++) {;
	  
	  int  bound = h_nzcnt[i];
	  for(int k=0;k<bound;k++ ) {
		int j = h_ptr[k] + i;
		int in = h_indices[j];
		float d = h_data[j];
		mat[h_perm[i]][in] = d;
		}
	}
	int nnz =0;
	convertToCSR(dim, mat, rowstr, colidx, values, &nnz);
    mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO, 1, 1, rowstr, rowstr+1, colidx, values);
  }

	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);


	
  int p, i, k;
	//main execution
	for(p=0;p<50;p++)
	{
		if (mkl) {
			double alpha = 1.0, beta = 0.0;
    struct matrix_descr descr = {SPARSE_MATRIX_TYPE_GENERAL};
    mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, descr, h_x_vector, beta, h_Ax_vector);
		}
		for (i = 0; i < dim; i++) {
		  float sum = 0.0f;
		  //int  bound = h_nzcnt[i / 32];
		  int  bound = h_nzcnt[i];
		  for(k=0;k<bound;k++ ) {
			int j = h_ptr[k] + i;
			int in = h_indices[j];

			float d = h_data[j];
			float t = h_x_vector[in];

			sum += d*t;
		  }
		  h_Ax_vector[h_perm[i]] = sum;
		}
	}	

	if (parameters->outFile) {
		pb_SwitchToTimer(&timers, pb_TimerID_IO);
		outputData(parameters->outFile,h_Ax_vector,dim);
		
	}
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	free (h_data);
	free (h_indices);
	free (h_ptr);
	free (h_perm);
	free (h_nzcnt);
	free (h_Ax_vector);
	free (h_x_vector);
	pb_SwitchToTimer(&timers, pb_TimerID_NONE);

	pb_PrintTimerSet(&timers);
	pb_FreeParameters(parameters);

	return 0;

}
