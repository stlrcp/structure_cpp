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
    return pre;
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
    print_Node(l3->next);
    return 0;
}
