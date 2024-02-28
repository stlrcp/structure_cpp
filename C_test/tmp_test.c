/*
// C/C++ 语言性能分析方法及性能分析工具的使用 --- https://blog.csdn.net/qq_41854911/article/details/122245100
#include <stdio.h>
void loop(int n){
    int m = 0;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            m++;
        }
    }
}
void fun2(){
    return;
}
void fun1(){
    fun2();
}
int main(){
    loop(10000);
    // fun1callfun2
    fun1();
    return 0;
}


// 整型加和减
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main(){
    clock_t begin, end;
    double cost;
    // 开始记录
    begin = clock();
    // 待测试程序段
    int a = 1;
    for (int i = 0; i < 100000000; i++){
        // a = a + 1;
        a = a - 1;
    }
    // 结束记录
    end = clock();
    cost = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("constant CLOCKS_PER_SEC is: %ld, time cost is: %lf secs\n", CLOCKS_PER_SEC, cost);
} // 整型乘 和 整型加和减也差不多，整型除 相比上面会更耗时，但量级差不多


// 浮点型加和减
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main(){
    clock_t begin, end;
    double cost;
    // 开始记录
    begin = clock();
    // 待测试程序段
    double a = 1.0;
    for (int i = 0; i < 100000000; i++){
        a = a + 1; // a = a -1;
    }
    // 结束记录
    end = clock();
    cost = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("constant CLOCKS_PER_SEC is: %ld, time cost is: %lf secs\n", CLOCKS_PER_SEC, cost);
    return 0;
} // 浮点型的加和减耗时大概是整型加减的 5 倍


// 浮点乘除：
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main(){
    clock_t begin, end;
    double cost;
    // 开始记录
    begin = clock();
    // 待测试程序段
    double a = 1.0;
    for (int i = 0; i < 100000000; i++){
        a = a / i;
    }
    // 结束记录
    end = clock();
    cost = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("constant CLOCKS_PER_SEC is: %ld, time cost is: %lf secs\n", CLOCKS_PER_SEC, cost);
    return 0;
} // 浮点型的乘和除耗时大概是浮点型的加和减耗时的 2 倍


//  测试打印 printf
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main(){
    clock_t begin, end;
    double cost;
    // 开始记录
    begin = clock();
    // 待测试程序段
    for (int i = 0; i < 1000; i++){
        printf("hello");
    }
    // 结束记录
    end = clock();
    cost = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("constant CLOCKS_PER_SEC is: %ld, time cost is: %lf secs\n", CLOCKS_PER_SEC, cost);
    return 0;
} // 打印语句耗时和打印的内容中字符的长短有关。差不多和字符的长度成正比。
*/

// 函数调用
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int callme(int a){
    a = a + 1;
    return a;
}
int main(){
    clock_t begin, end;
    double cost;
    // 开始记录
    begin = clock();
    // 待测试程序段
    int b;
    for (int i = 0; i < 100000000; i++){
        b = callme(i);
    }
    // 结束记录
    end = clock();
    cost = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("constant CLOCKS_PER_SEC is: %ld, time cost is: %lf secs\n", CLOCKS_PER_SEC, cost);
    return 0;
} // C语言中，我们会把经常调用的代码不长的函数弄成宏或内联函数，这样可以提高运行效率
