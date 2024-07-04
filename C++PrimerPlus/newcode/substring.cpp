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


// 代码实现 - 内部 cout 主要是为了方便 debug
#include <iostream>
#include <cstring>
using namespace std;

// 截取部分子串 https://docs.pingcode.com/ask/ask-ask/257780.html?p=257780
void sub_str(char* source, int begin, int length, char* target){
    // cout << source << endl;
    // cout << target << endl;
    // cout << begin << endl;
    // cout << length << endl;
    int i;
    for (i=0; i<length; i++){
        target[i] = source[begin+i];
        // cout << target << endl;
    }
    target[i] = '\0';      //  必须加，不然会出现内存被占用，得到的 target 长度异常
}

// 检查短子串是否存在于长子串内部 
bool check_sub(char* l_str, char* s_str){
    char l_s_str[100] = "";
    bool equal;
    for (int i=0; i<(strlen(l_str)-strlen(s_str)+1); i++){
        for(int j=0; j<strlen(s_str); j++){
            l_s_str[j] = l_str[i+j];
        }
        // cout << l_s_str << endl;
        // if (l_s_str == s_str)
        //     return true;
        
        for (int n=0; n<strlen(s_str); n++){
            if(s_str[n] != l_s_str[n]){
                break;
            }
            if (n == (strlen(s_str)-1))  
                return true; 
        }
    }
    return false;

}

int main(){
    char* A ="abczhdshoozjpzjoppzhiahsopppp";
    char* B ="shsojosmxhposhoozjapsjnxjhp";
    cout << "A len = " << strlen(A) << endl;
    cout << "B len = " << strlen(B) << endl;
    if (strlen(A) > strlen(B)){
        char* tmp = A;
        A = B;
        B = tmp;
    }
    char tmp_con[100] = "";
    char tmp_str[100] = "";    // 必须初始化，不然内存污染，打印出的tmp_str异常
    for (int i=0; i<strlen(A); i++){
        for (int j=0; j<(strlen(A)-i); j++){
            sub_str(A, i, j+1, tmp_str);
            // cout << "i = " << i << "  " << tmp_str << endl;
            if(check_sub(B, tmp_str) && strlen(tmp_str) > strlen(tmp_con)){
                cout << strlen(tmp_str) << endl;
                sub_str(tmp_str, 0, strlen(tmp_str), tmp_con);
            }
                

            // tmp_str = tmp_con;
        }
        for (int n=0; n<strlen(tmp_str); n++){
               tmp_str[n] = '\0';
        }
        // break;
        // cout << strlen(tmp_str) << endl;
        // cout << tmp_str << endl;
    }
    cout << tmp_con << endl;
    // cout << strlen(tmp_str) << endl;
    return 0;
}


// 简化代码如下
#include <iostream>
#include <cstring>
using namespace std;

void sub_str(char* source, int begin, int length, char* target){
    int i;
    for (i=0; i<length; i++){
        target[i] = source[begin+i];
    }
    target[i] = '\0';      //  必须加，不然会出现内存被占用，得到的 target 长度异常
}

bool check_sub(char* l_str, char* s_str){
    char l_s_str[100] = "";
    bool equal;
    for (int i=0; i<(strlen(l_str)-strlen(s_str)+1); i++){
        for(int j=0; j<strlen(s_str); j++){
            l_s_str[j] = l_str[i+j];
        }
        for (int n=0; n<strlen(s_str); n++){
            if(s_str[n] != l_s_str[n]){
                break;
            }
            if (n == (strlen(s_str)-1))  
                return true; 
        }
    }
    return false;
}

int main(){
    char* A ="abczhdshoozjpzjoppzhiahsopppp";
    char* B ="shsojosmxhposhoozjapsjnxjhp";
    if (strlen(A) > strlen(B)){
        char* tmp = A;
        A = B;
        B = tmp;
    }
    char tmp_con[100] = "";
    char tmp_str[100] = "";    // 必须初始化，不然内存污染，打印出的tmp_str异常
    for (int i=0; i<strlen(A); i++){
        for (int j=0; j<(strlen(A)-i); j++){
            sub_str(A, i, j+1, tmp_str);
            if(check_sub(B, tmp_str) && strlen(tmp_str) > strlen(tmp_con)){
                sub_str(tmp_str, 0, strlen(tmp_str), tmp_con);
            }
        }
        for (int n=0; n<strlen(tmp_str); n++){
               tmp_str[n] = '\0';
        }
    }
    cout << tmp_con << endl;
    return 0;
}


// 动态规划？
#include <bits/stdc++.h>
using namespace std;
string LCS(string str1, string str2){
    if(str1.size() > str2.size())
        swap(str1, str2);
    int m = str1.size();
    int n = str2.size();
    // dp[i][j] str1前i个字符和str2前j个字符(以其为尾字符)的最长公共子串长度
    int dp[m+1][n+1];
    int maxlen=0, end=0;
    // base case
    for(int i=0; i<=m; ++i) dp[i][0] =0;
    for(int j=0; j<=n; ++j) dp[0][j] = 0;
    for(int i=1; i<=m; ++i){
        for(int j=1; j<=n; ++j){
            if(str1[i-1] == str2[j-1])
                dp[i][j] = dp[i-1][j-1] + 1;
            else
                dp[i][j] = 0;
            if(dp[i][j] > maxlen){
                maxlen=dp[i][j];
                end = i-1;
            }
        }
    }
    if(maxlen == 0) return "-1";
    else
        return str1.substr(end-maxlen+1, maxlen);
}
int main(){
    string s1, s2;
    while(cin>>s1>>s2)
        cout << LCS(s1, s2) << endl;
}


// 暴力寻找
#include <iostream>
#include <vector>
#include <cstring>
using namespace std;
string a, b, minn = "";
string cut(int l, int r){
    string tmp="";
    for(int i=l; i<=r; i++)
        tmp += a[i];
    return tmp;  // l 代表左端点，r代表右端点
}
void solve(){
    if (a.size() > b.size())
        swap(a, b);
    for(int i=0; i<a.size(); i++){
        for(int j=i; j<a.size(); j++){
            string tmp = cut(i, j);
            if(b.find(tmp) != string::npos)
                if(tmp.size() > minn.size())
                    minn =tmp;
        }
    }
    cout << minn << "\n";
}
signed main(){
    while(cin>>a>>b){
        minn = ""; // 因为有多组输入，这里进行一个清空的操作
        solve();
    }
    return 0;
}


// 主要使用了 string 库中的方法
#include <iostream>
#include <string>
#include <algorithm>
using namespace std;
int main(){
    string s1, s2;
    while(cin >> s1>>s2){
        if(s1.length() > s2.length())
            swap(s1, s2);
        string output = "";
        for(int i=0; i<s1.length(); i++){
            for(int j=i; j<s1.length(); j++){
                if(int(s2.find(s1.substr(i, j-i+1))) < 0)
                    break;
                else if(output.length() < j-i+1) // 更新较长的子串
                    output = s1.substr(i, j-i+1); 
            }
        }
        cout << output << endl;
    }
    return 0;
}


// 枚举改进 - 类似于动态规划
#include <iostream>
#include <string>
#include <algorithm>
using namespace std;
int main(){
    string s1, s2;
    while(cin>>s1>>s2){
        if(s1.length() > s2.length())
            swap(s1, s2);
        string output="";
        for(int i=0; i<s1.length(); i++){
            for(int j=0; j<s2.length(); j++){
                int length =0;
                int x=i, y=j;
                while(x<s1.length() && y<s2.length() && s1[x] == s2[y]){
                    x++;
                    y++;
                    length++;
                }
                if(output.length() < length)
                    output = s1.substr(i, x-i);
            }
        }
        cout << output << endl;
    }
    return 0;
}
*/

// 动态规划
#include <iostream>
#include <string>
#include <vector>
using namespace std;
int main(){
    string s1, s2;
    while(cin>>s1>>s2){
        if(s1.length() > s2.length())
            swap(s1, s2);
        vector<vector<int>> dp(s1.length()+1, vector<int>(s2.length()+1, 0)); // dp[i][j] 表示到s1第i个个到s2第j个为止的公共子串长度
        int max=0, end=0;
        for(int i=1; i<s1.length(); i++){
            for(int j=1; j<=s2.length(); j++){
                if(s1[i-1] == s2[j-1])
                    dp[i][j] = dp[i-1][j-1] + 1;
                else
                    dp[i][j] = 0;
                if(dp[i][j] > max){
                    max = dp[i][j];
                    end = i-1;
                }
            }
        }
        cout << s1.substr(end-max+1, max) << endl;
    }
    return 0;
}
