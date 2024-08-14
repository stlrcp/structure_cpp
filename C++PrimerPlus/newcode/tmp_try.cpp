/*
#include <iostream>
using namespace std;
class Time{
    public:
        Time(int hour = 12){
            _hour = 24;
            _hour = 34;
        };
        void Printf(){
            cout << "hour为： " << _hour << endl;
        }
    private : 
        int _hour;
};
int main()
{
    Time t1;
    t1.Printf();
    return 0;
}


#include <iostream>
using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr){}
};

void addNode(ListNode *in_node, int num){
    ListNode *out_node = in_node;
    ListNode *tmp = new ListNode(num);
    while(out_node->next != nullptr)
        out_node = out_node->next;
    out_node->next = tmp;
}

void print_Node(ListNode *in_node){
    ListNode *out_node = in_node;
    do
    {
        printf("======= the val = %d\n", out_node->val);
        out_node = out_node->next;
    } while (out_node != nullptr);
}

ListNode* addList(ListNode *l1, ListNode *l2){
    ListNode *l1_t = l1;
    ListNode *l2_t = l2;
    ListNode *l3;
    ListNode *pre = l3;
    int res = 0;
    int tmp, sum, l1_val, l2_val;
    while (l1_t != nullptr || l2_t != nullptr || res != 0){
        printf("== l1_val = %d, l2_val = %d, res = %d\n", l1_val, l2_val, res);
        if (l1_t != nullptr)
        {
            l1_val = l1_t->val;
            l1_t = l1_t->next;
        }
        else
        {
            l1_val = 0;
        }
        if(l2_t != nullptr){
            l2_val = l2_t->val;
            l2_t = l2_t->next;
        }else{
            l2_val = 0;
        }
        sum = res + l1_val + l2_val;
        ListNode *tmp = new ListNode(sum % 10);
        l3->next = tmp;
        l3 = l3->next;
        res = sum / 10;
    }
    return pre->next;
}

int main(){
    ListNode *l1 = new ListNode(-1);
    addNode(l1, 0);
    addNode(l1, 1);
    addNode(l1, 2);
    addNode(l1, 3);
    addNode(l1, 4);
    ListNode *l2 = new ListNode(6);
    addNode(l2, 7);
    addNode(l2, 8);
    addNode(l2, 9);
    addNode(l2, 1);
    ListNode *l3;
    l3 = addList(l1, l2);
    print_Node(l3);
    return 0;
}


#include <iostream>
#include <string>
using namespace std;
string tmp = " the fist input tmpahaajojoj";
int main(){
    cout << tmp << endl;
    int num = 0;
    for (char i : tmp)
    {
        if (i == ' ')
            num = 0;
        else
            num++;
    }
    cout << num << endl;
    return 0;
}


#include <iostream>
#include <string>
using namespace std;
string tmp = "tmpzh angsu ibiAHSSOH  OJ123679001524646182979  ";
// char b = 's';  // s = 115, S = 83
char b = '6';
int main()
{
    cout << tmp << endl;
    // cout << int(b) << endl;
    // cout << char(115) << endl;
    auto b_after = (b >= 'A' && b <= 'Z') ? int(b) + 32 : int(b);
    // cout << b_after << endl;
    int a_after;
    int num = 0;
    for (char i : tmp)
    {
        // cout << int(i) << endl;
        if (i >= 'A' && i <= 'Z')
            a_after = int(i) + 32;
        else
            a_after = int(i);
        if (a_after == b_after)
            num++;
    }
    cout << num << endl;
    return 0;
}


#include <iostream>
#include <set>
using namespace std;
int tmp[15] = {2, 7, 6, 3, 4, 5, 7, 2, 2, 3, 8, 10, 5, 4, 2};
// set tmp = {2, 7, 6, 3, 4, 5, 7, 2, 2, 3, 8, 10, 5, 4, 2};
int main(){
    set<int> res;
    for (auto i : tmp)
    {
        res.insert(i);
    }
    for(auto i : res){
        cout << i << endl;
    }
    return 0;
}


#include <iostream>
using namespace std;

void cout_rank(int num){
    int num_list[1001] = {0};
    int random;
    while (num)
    {
        cin >> random;
        num_list[random] = random;
        num--;
    }
    for (int i = 0; i < 1001; i++){
        if(num_list[i] != 0)
            cout << num_list[i] << endl;
    }
}
int main(){
    int num;
    cin >> num;
    cout_rank(num);
}


#include <iostream>
#include <string>
using namespace std;
string tmp = "tmpzh angsu ibiAHSSOH  OJ123679001524646182979  zj";
int main()
{
    cout << tmp << endl;
    cout << tmp.length() << endl;
    // string t_1 = tmp.substr(0, 8);
    int t_l;
    t_l = tmp.length() / 8;
    t_l = (tmp.length() % 8) ? t_l + 1 : t_l;
    cout << t_l << endl;
    string t_m = "";
    for (int i = 0; i < t_l; i++)
    {
        if (i != t_l-1)
            cout << tmp.substr(i * 8, 8)<< endl;
        else{
            for (int j = 0; j < (8 - tmp.substr(i * 8, 8).length()); j++){
                t_m = t_m + "0";
            }
            cout << tmp.substr(i * 8, 8) + t_m << endl;
        }
    }
    // string a = "zh";
    // string b = "0";
    // cout << a + b << endl;
}


#include <iostream>
#include <string>
using namespace std;
int main(){
    string str;
    while(cin >> str){
        int len = str.size();
        if(len % 8 != 0){
            int count = 8 - len % 8;
            str.append(count, '0');
        }
        int newLen = str.size();
        for (int i = 0; i < newLen; i+=8){
            cout << str.substr(i, 8) << endl;
        }
    }
    return 0;
}


#include <iostream>
#include <string>
#include <map>
#include <cmath>
using namespace std;
int main(){
    map<char, int> t_m = {{'1',1}, {'a', 10}, {'b', 11}, {'c', 12}, {'d', 13}, {'e', 14}, {'f', 15},
    {'2', 2}, {'3',3},{'4',4},{'5',5},{'6',6},{'7',7},{'8',8},{'9',9}};
    // string a = "0x12";
    string a = "0x12bacf";
    // cout << a.substr(2).size() << endl;
    // string b = "e";
    // cout << t_m.at(b) << endl;
    int len, res;
    len = a.substr(2).size() - 1;
    for (auto i : a.substr(2))
    {
        // cout << t_m.at(i) << endl;
        res += t_m.at(i) * pow(16, len);
        len--;
    }
    // int a_i = 2;
    // cout << pow(a_i, 2) << endl;
    cout << res << endl;
}
*/
