void kernel(float A[12][11], float s[11], float q[12], float p[11], float r[12]) {
    int i;
    int j;

    loop_1: for (i = 0; i < 11; i++) {
        s[i] = 0.0f;
    }

    loop_2: for (i = 0; i < 12; i++) {
        q[i] = 0.0f;
    }

    loop_3: for (j = 0; j < 11; j++) {
        float acc = 0.0f;

        loop_4: for (i = 0; i < 12; i++) {
            acc += r[i] * A[i][j];
        }

        s[j] = acc;
    }

    loop_5: for (i = 0; i < 12; i++) {
        float acc2 = 0.0f;

        loop_6: for (j = 0; j < 11; j++) {
            acc2 += A[i][j] * p[j];
        }
        
        q[i] = acc2;
    }
}
