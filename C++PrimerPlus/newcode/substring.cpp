/*
#include <iostream>
#include <cstring>
using namespace std;

int main(){
    // 测试是否可以使用 == 来判断相等
    char* t_a = "aaaa";
    char* t_b = "aaaa";
    if (t_a == t_b)
        cout << "t_a = t_b" << endl;
  
    // 测试 char[] 初始化的能否赋值, 已知 char* 初始化的赋值会报错
    char t_c[300];
    t_c[0] = t_a[0];
    cout << t_c << t_a << endl;

    // 判断 char* 和 char[] 初始化的是否相等
    char* t_a = "aaaa";
    char t_b[10] = "aaaa";
    if(t_a == t_b)
        cout << "t_a == t_b" << endl;
    for(int i=0; i<strlen(t_a); i++){
        if(t_a[i] == t_b[i])
            cout << "the " << i << " index is queal" << endl;
    } 

    // 尝试证明 t_a 在 t_b 的方法
    char* t_a = "aaaab";
    char t_b[10] = "aaaa";
    if(t_a == t_b)
        cout << "t_a == t_b" << endl;
    for(int i=0; i<strlen(t_a); i++){
        if(t_a[i] != t_b[i])
            break;
        if (i == (strlen(t_a)-1))
            cout << " can prove t_a equal t_b" << endl; 
    }

    // 使用自己编写的 check_sub 方法判断 s_a 是否存在 a 内部
    char* a = "abceeeed";
    char* s_a = "ceeb";
    if(check_sub(a, s_a))
        cout << " true in " << endl;

    // 针对 char[] 初始化的对象，进行赋值，赋值'a'成功，赋值"a"失败，在C++中注意区分单引号和双引号
    char tmp_s[100]="";
    for (int i=0; i<10; i++){
        tmp_s[i] = 'a';
    } 
    cout << strlen(tmp_s) << endl;
    for (int j=0; j<strlen(tmp_s); j++){
        tmp_s[j] = '\0';     // 将已经赋值的 char[] 对象重新初始化，这种方式不彻底，'\0'只是表明在此结束，并未改写内存中数值
    }
    cout << strlen(tmp_s) << endl;

    // 使用自己编写的 sub_str 提取部分字串方法
    char tmp_a[] = "abazjjx";
    cout << tmp_a << endl;
    char *tmp_b = "abcde";
    sub_str(tmp_b, 0, strlen(tmp_b), tmp_a);
    cout << tmp_a << endl;

    return 0;
}
*/
