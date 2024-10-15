/*
// 九宫格
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

// 输入
// 0 9 2 4 8 1 7 6 3
// 4 1 3 7 6 2 9 8 5
// 8 6 7 3 5 9 4 1 2
// 6 2 4 1 9 5 3 7 8
// 7 5 9 8 4 3 1 2 6
// 1 3 8 6 2 7 5 9 4
// 2 7 1 5 3 8 6 4 9
// 3 8 6 9 1 4 2 5 7
// 0 4 5 2 7 6 8 3 1
*/


// 有效的括号
#include <iostream>
#include <string>
#include <vector>
using namespace std;
bool is_valid(string input){
    vector<char> cha_vec;
    for (auto i : input)
    {
        if (i == '{' || i == '(' || i == '[' )
            cha_vec.push_back(i);
        else if (i == '}' && cha_vec.back() == '{')
            cha_vec.pop_back();
        else if (i == ']' && cha_vec.back() == '[')
            cha_vec.pop_back();
        else if (i == ')' && cha_vec.back() == '(')
            cha_vec.pop_back();
        else
            return false;
    }
    if (cha_vec.size() > 0){
        return false;
    }
    return true;
}
int main(){
    string inp = "{[({({}{{[[]]}}())})]}{{";
    if (is_valid(inp))
        cout << "success" << endl;
    else
        cout << "failed" << endl;
}
