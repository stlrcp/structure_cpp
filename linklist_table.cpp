#include <iostream>
using namespace std;

typedef int ElemType;

// 链表的定义
struct Lnode
{
    ElemType data; // 节点的数据域
    Lnode *next;  // 节点的指针域
};
typedef Lnode *LinkList;

// 链表的初始化
bool InitList(LinkList &L)  // 题外话： LinkList &L 等价于 Lnode *&L, Lnode *&L 是一个指向指针的引用
{
    L = new Lnode;  // 堆区开辟一个头节点， 节点的数据类型为Lnode
    L->next = nullptr;  // 空表，也就是说头节点的指针指向为空
    return true;
}

// 头插法创建单向链表
void CreateListHead(LinkList &L, const size_t n)
{
    for (int i = 0; i < n; ++i)
    {
        Lnode *p = new Lnode;
        cin >> p->data;
        p->next = L->next;
        L->next = p;
    }
}

int main()
{
    Lnode *L;
    int n = 3;
    InitList(L);
    CreateListHead(L, n);
    cout << L << endl;
    cout << L->next << endl;
    cout << L->data << endl;
    return 0;
}
