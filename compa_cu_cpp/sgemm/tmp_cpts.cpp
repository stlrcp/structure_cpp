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
    for (int i = 0; i < width; i++){
        // copy t_a
        for (int c_i = 0; c_i < 4; c_i++){
            if(i % 2 == 0)
                t_a[c_i] = a[(i+(c_i/2))* 4 + c_i%2];
            else
                t_a[c_i] = a[(i+(c_i/2))*4 + c_i%2 - 2];
        }
        // copy t_b
        for (int b_i = 0; b_i < 4; b_i++){
            if(i % 2 == 0)
                t_b[b_i] = b[(i + (b_i/2) - i/2) * 4 + b_i%2 - (i/2)*2];
            else
                t_b[b_i] = b[(i+1 + (b_i / 2) -i/2) * 4 + b_i % 2- (i/2)*2];
        }
        // sub_matrix_mul
        for (int b_i = 0; b_i < 2; b_i++)
        {
            for (int b_j = 0; b_j < 2; b_j++)
            {
                float tsum = 0.0;
                for (int b_k = 0; b_k < 2; b_k++)
                    tsum += t_a[b_i * 2 + b_k] * t_b[b_k * 2 + b_j];
                t_c[b_i * 2 + b_j] = tsum;
            }
        }
        // copy part t_c
        for(int c_i =0; c_i < 4; c_i++){
            c[(i/2 +(c_i/2))*4 + c_i%2 + (i/2)*6] += t_c[c_i];
            // 不等同于 c[(i +(c_i/2))*4 + c_i%2 + i] += t_c[c_i]; 内部不能够化简，因为内部存在隐式的取整操作 类似下面的 3/2*2 = 2 的问题
        }
        // copy t_a
        //     1  2  3  4          1  2  5  6
        //     5  6  7  8          3  4  7  8
        //     9 10 11 12    ->    9 10 13 14
        //    13 14 15 16         11 12 15 16
        for (int c_i = 0; c_i < 4; c_i++){
            if(i % 2 == 0)
                t_a[c_i] = a[(i+(c_i/2))* 4 + c_i%2];
            else
                t_a[c_i] = a[(i+(c_i/2))*4 + c_i%2 - 2];
        }
        // copy t_b
        for (int b_i = 0; b_i < 4; b_i++){
            if(i/2 == 0)
                t_b[b_i] = b[(2*i+b_i/2) * 4 + b_i % 2 + 2];
            else
                t_b[b_i] = b[((i%2)*2 + b_i/2)*4 + b_i%2];
        }
        // sub_matrix_mul
        for (int b_i = 0; b_i < 2; b_i++)
        {
            for (int b_j = 0; b_j < 2; b_j++)
            {
                float tsum = 0.0;
                for (int b_k = 0; b_k < 2; b_k++)
                    tsum += t_a[b_i * 2 + b_k] * t_b[b_k * 2 + b_j];
                t_c[b_i * 2 + b_j] = tsum;
            }
        }
        for(int c_i =0; c_i < 4; c_i++){
            if(i/2 == 0)
                c[(i/2 + (c_i/2))*4 + c_i%2 + 2] += t_c[c_i];
            else
                c[((i/2)*2 + (c_i / 2)) * 4 + c_i % 2] += t_c[c_i];
        }

        for (int c_i = 0; c_i < 4; c_i++)
            cout << setw(3) << t_c[c_i] << " ";
        cout << endl;
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
        // b[i] = (i / width) + 1;
        b[i] = i + 1;
    }
    gmcpu(a, b, c, width);
    for (int i = 0; i < width * width; i++)
    {
        cout << c[i] << " ";
    }
    cout << endl;

    // for (int i = 0; i < width; i++){
    //     float c_a[4] = {0};
    //     float c_b[4] = {0};
    //     for (int c_i = 0; c_i < 4; c_i++){
    //         if(i%2 == 0)
    //             c_a[c_i] = a[(i+(c_i/2))* 4 + c_i%2];
    //         else
    //             c_a[c_i] = a[(i+(c_i/2))*4 + c_i%2 - 2];
    //     }
    //     for (int b_i = 0; b_i < 4; b_i++){
    //         if(i%2 == 0)
    //             // c_b[b_i] = b[(i%2 + (i) + (b_i / 2)) * 4 + b_i % 2];
    //             c_b[b_i] = b[(i + (b_i/2) - i/2) * 4 + b_i%2 - (i/2)*2];
    //         else
    //             // c_b[b_i] = b[(i%2 + (i) +  (b_i / 2)) * 4 + (b_i % 2)+2];
    //             c_b[b_i] = b[(i+1 + (b_i / 2) -i/2) * 4 + b_i % 2- (i/2)*2];
    //     }

    //     // for (int t_j = 0; t_j < 4; t_j++)
    //     //     cout << setw(3) << c_a[t_j] << " ";
    //     cout << endl;
    //     for (int t_j = 0; t_j < 4; t_j++)
    //         cout << setw(3) << c_b[t_j] << " ";
    //     cout << endl;
    // }

    for (int i = 0; i < width*width; i++){
        c[i] = 0;
    }
    sgmcpu(a, b, c, width);
    // int t_m = 3;    // 简单小测试在 cpp 中，/ 除，同时结合 int 数据特性
    // cout << t_m / 2 * 2 << endl;  // 答案为 2
    for (int i = 0; i < width * width; i++)
    {
        cout << setw(3) << c[i] << " ";
        if(i % width==3)
            cout << endl;
    }
    cout << endl;
    return 0;
}
