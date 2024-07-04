/*
//  测试运算符的优先级 || 和 && 连续使用的理解
#include <iostream>
using namespace std;
int main(){
    int a = 10;
    int b = 10;
    int c = 8;
    int d = 8;
    if ((a != b || a != c) && a > c){
        cout << "the first is true and second is true" << endl;
    }
    return 0;
}

// 测试 ++i 在 for 循环中的影响
#include <iostream>
using namespace std;
int main(){
    for(int i =0; i<10; ++i)
    {
        cout << "i = " << i << endl;
    }
    return 0;
}
*/


#include <iostream>
using namespace std;

int num[9][9];
bool flag = false;

bool check(int index){
    int row = index / 9;
    int col = index % 9;
    for(int i=0; i<9; i++){
        if( i != col && num[row][i] == num[row][col]){
            return false;
        }
    }
    for(int j=0; j<9; j++){
        if( j != row && num[j][col] == num[row][col]){
            return false;
        }
    }
    for (int i=row/3*3; i<row/3*3+3; i++){
        for(int j=col/3*3; j<col/3*3+3; j++){
            if((i!=row || j != col) && num[i][j] == num[row][col]){
                return false;
            }
        }
    }
    return true;
}

void dfs(int n){
    if (n == 81){
        for (int i=0; i<9; i++){
            for(int j=0; j<8; j++){
                cout << num[i][j] << " ";
            }
            cout << num[i][8] << endl;
        }
        flag = true;
        return;
    }
    int h = n / 9;
    int l = n % 9;
    // cout << "====== " << n << endl;
    if(num[h][l] == 0){
        cout << "==== num[h][l] = " << num[h][l] << endl;
        for(int i=1; i<=9; i++){
            num[h][l] = i;
            if(check(n)){
                dfs(n+1);
                if(flag){
                    return;
                }
            }
        }
        // num[h][l] = 0;
    }else{
        dfs(n+1);
    }
}

int main(){
    for(int i=0; i<9; ++i){
        for(int j=0; j<9; ++j){
            cin >> num[i][j];
        }
    }
    dfs(0);

    // for (int i=0; i<9; ++i){
    //     for(int j=0; j<8; ++j){
    //         cout << num[i][j] << " ";
    //     }
    //     cout << num[i][8] << endl;
    // }
    return 0;
}
