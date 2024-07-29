/*
#include <iostream>
using namespace std;
#define OFFSET(row, col, ld) ((row)*(ld)+(col))
void cpuSgemm(float *a, float *b, float *c, const int M, const int N, const int K){
    for(int m=0; m<M; m++){
        for(int n=0; n<N; n++){
            float psum = 0.0;   //  消除原始地址上的值
            for(int k=0; k<K; k++){
                psum += a[OFFSET(m,k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}
int main(){
    const int M = 2, N=2, K = 3;
    size_t size_a = M*K*sizeof(float);
    size_t size_b = K*N*sizeof(float);
    size_t size_c = M*N*sizeof(float);
    float *h_a, *h_b, *h_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    for (int i=0; i< M; i++){
        for(int j=0; j<K; j++){
            h_a[K*i + j] = K*i+j;
        }
    }

    for(int i=0; i<K; i++){
        for(int j=0; j<N; j++){
            h_b[N*i+j] = N*i+j;
            cout << N*i+j << endl;
        }
    }
    cpuSgemm(h_a, h_b, h_c, M, N, K);
    cout << h_c[0] << endl;
    cout << h_c[1] << endl;
    cout << h_c[2] << endl;
    cout << h_c[3] << endl;
    return 0;
}
*/

#include <iostream>
using namespace std;

void cpuGemm(float *a, float *b, float *c, int width){
    for(int i=0; i<width; i++){
        for(int j=0; j<width; j++){
            float sum = 0.0f;
            for(int k=0; k<width; k++){
                sum += a[i*width+k] * b[k*width+j];
            }
            c[i*width+j] = sum;
        }
    }
}

void matrixMulCpu(float* A, float* B, float* C, int width){
    float sum = 0.0f;
    for(int i = 0; i < width; i++){
        for(int j = 0; j < width; j++){
            for(int l = 0; l < width; l++){
                sum += A[i * width + l] * B[l * width + j];
            }
            C[i * width + j] = sum;
            sum = 0.0f;
        }
    }
}

int main(){
    int width=3;
    float *h_a, *h_b, *h_c;
    size_t size_a = width*width*sizeof(float);
    size_t size_b = width*width*sizeof(float);
    size_t size_c = width*width*sizeof(float);
    h_a = (float*)malloc(size_a);
    h_b = (float*)malloc(size_b);
    h_c = (float*)malloc(size_c);
    for(int i=0; i<9; i++){
        h_a[i] = 1;
        h_b[i] = i+1;
    }
    cpuGemm(h_a, h_b, h_c, width);
    for(int i=0; i<9; i++)
        cout << h_c[i] << endl;
    return 0;
}
