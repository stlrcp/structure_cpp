#include <iostream>
#include <iomanip>
using namespace std;

void gmcpu(float *a, float *b, float *c, int width){
    for (int i = 0; i < width; i++){
        for (int j = 0; j < width; j++){
            float psum = 0.0;
            for (int k = 0; k < width; k++){
                psum += a[i * width + k] * b[k * width + j];
            }
            c[i * width + j] = psum;
        }
    }
}

void sgmcpu(float *a, float *b, float *c, int width){
    int b_off = 2;
    float *t_a, *t_b, *t_c;
    size_t t_size = sizeof(float) * b_off * b_off;
    t_a = (float *)malloc(t_size);
    t_b = (float *)malloc(t_size);
    t_c = (float *)malloc(t_size);
    for (int i = 0; i < width *width; i++)
        // copy t_a
        for (int c_i = 0; c_i < 4; c_i++){
            if(c_i<b_off)
                t_a[c_i] = a[i];
            else
                t_a[c_i] = a[c_i * width];
        }

            for (int b_i = 0; b_i < 2; b_i++)
            {
                for (int b_j = 0; b_j < 2; b_j++)
                {
                    float tsum = 0.0;
                    for (int b_k = 0; b_k < 2; b_k++)
                        tsum += a[b_i * 2 + b_k] * b[b_k * 2 + b_j];
                    c[b_i * 2 + b_j] = tsum;
                }
            }
}

int main(){
    int width = 4;
    float *a, *b, *c;
    size_t f_size = sizeof(float) * width * width;
    a = (float*)malloc(f_size);
    b = (float*)malloc(f_size);
    c = (float*)malloc(f_size);
    for (int i = 0; i < width*width; i++){
        a[i] = i + 1;
        b[i] = (i / width) + 1;
    }
    gmcpu(a, b, c, width);
    // for (int i = 0; i < width * width; i++)
    // {
    //     cout << c[i] << " ";
    // }
    cout << endl;

    for (int i = 0; i < width; i++){
        float c_a[4] = {0};
        for (int c_i = 0; c_i < 4; c_i++){
            c_a[c_i] = a[(i+(c_i/2))* 4 + c_i%2];
        }

        for (int t_j = 0; t_j < 4; t_j++)
            cout << setw(3) << c_a[t_j] << " ";
        cout << endl;
    }
    return 0;
}
