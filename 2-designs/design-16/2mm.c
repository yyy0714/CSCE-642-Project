void kernel(float alpha, float beta, float tmp[4][5], float A[4][7], float B[7][5], float C[5][8], float D[4][8]) {
    int i;
    int j;
    int k;
    
    loop_1: for (i = 0; i < 4; i++) {
        loop_2: for (j = 0; j < 5; j++) {
            tmp[i][j] = 0.0;

            loop_3: for (k = 0; k < 7; ++k) {
                tmp[i][j] += alpha * A[i][k] * B[k][j];
            }
        }
    }

    loop_4: for (i = 0; i < 4; i++) {
        loop_5: for (j = 0; j < 8; j++) {
            D[i][j] *= beta;

            loop_6: for (k = 0; k < 5; ++k) {
                D[i][j] += tmp[i][k] * C[k][j];
            }
        }
    }
}
