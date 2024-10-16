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



#include <iostream>
#include <string>
#include <vector>
using namespace std;

bool is_hw(string in_str){
    string tmp = "";
    for (int i = in_str.length() - 1; i >= 0; i--)
        tmp += in_str[i];
    if (in_str == tmp)
        return true;
    else
        return false;
}

int main(){
    string tmp = "aababjsllsssojoaql";
    // string tmp = "aabaa";
    vector<int> len;
    for (int i = 0; i < tmp.length(); i++)
    {
        for (int j = 0; j < tmp.length() - i; j++){
            string sub_tmp = tmp.substr(i, j+1);
            if (is_hw(sub_tmp)){
                len.push_back(sub_tmp.length());
                cout << "length = " << sub_tmp.length() << " sub_tmp = " << sub_tmp << endl;
            }
        }
    }
    return 0;
}
*/

#include <iostream>
using namespace std;

struct listNode{
    int num;
    listNode *next;
    listNode(int val) : num(val), next(nullptr) {}  // 必须有构造函数，方便 new 实现，不然要使用 malloc 分配内存
};

void print_list(listNode *tmp){
    while (tmp != nullptr)
    {
        cout << tmp->num << endl;

        tmp = tmp->next;
    }
}

listNode* delete_ind(listNode *tmp, int value){
    listNode *aft = new listNode(0);   // 初始化头
    listNode *pre_aft;
    cout << "aft = " << aft << endl;
    pre_aft = aft;     // 重命名获取首地址
    while (tmp->next != nullptr)    // 需要保持一致，不能越界
    {
        
        aft->next = tmp;
        if (tmp->next->num != value)   // 保证移除的是需要的
        {
            tmp = tmp->next;
        }
        else{
            cout << "tmp.num = " << tmp->num << endl;
            tmp = tmp->next->next;
        }
        aft = aft->next;
        }
    return pre_aft->next->next;
}

int main(){
    listNode *pre;
    listNode *head = new listNode(0);
    pre = head;
    // pre = &tmp;
    int nums[6] = {2, 4, 6, 3, 1, 5};
    for (auto i : nums){
        head->next = new listNode(i);
        // tmp.num = i;
        head = head->next;
    }
    print_list(pre);
    listNode *suff;
    suff = delete_ind(pre, 3);
    print_list(suff);
    return 0;
}
