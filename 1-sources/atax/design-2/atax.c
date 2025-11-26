void kernel(float A[11][12], float x[12], float y[12], float tmp[11]) {
    int i;
    int j;

    loop_1: for (i = 0; i < 12; i++) {
        y[i] = 0.0;
    }

    loop_2: for (i = 0; i < 11; i++) {
        tmp[i] = 0.0;

        loop_3: for (j = 0; j < 12; j++) {
            tmp[i] = tmp[i] + A[i][j] * x[j];
        }

        loop_4: for (j = 0; j < 12; j++) {
            y[j] = y[j] + A[i][j] * tmp[i];
        }
    }
}
