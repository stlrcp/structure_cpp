#include <iostream>
using namespace std;

typedef int ElemType;
#define MAXSIZE 100

// 线性表的定义
struct SqList
{
    ElemType *elem;   // 顺序线性表的表头
    int length;  // 顺序线性表的长度
};

// 线性表的初始化
bool InitList(SqList &L)
{
    L.elem = new ElemType[MAXSIZE];   // 在堆区开辟内存
    if (!L.elem)
    {
        cerr << "error" << endl;
        return false;
    }
    L.length = 10;  // 设定线性表长度为0
    return 1;
}

// 线性表的销毁
void DestroyList(SqList &L)
{
    if (L.elem)
    {
        delete L.elem;
    }
}

// 线性表的清空
void ClearList(SqList &L)
{
    L.length = 0;
}

// 判断线性表是否为空
bool IsEmpty(const SqList &L)
{
    return static_cast<bool>(L.length);
}

// 线性表的取值
bool GetElem(const SqList &L, const size_t i, ElemType &e)
{
    if (i<1 || i>MAXSIZE)
    {
        cerr << "out of range" << endl;
        return false;
    }
    e = L.elem[i - 1];
    return true;
}

// 线性表的查找
int LocateList(const SqList &L, const ElemType &e)
{
    for (int i = 0; i < L.length; ++i)
    {
        if (L.elem[i] == e)
        {
            return i + 1;  // 查找成功，返回其查找元素的第一个下标值
        }
    }
    return 0;  // 未能找到对应元素，返回0
    // 算法时间复杂度：O(n)
}

// 线性表的删除
bool EraseList(SqList &L, const int &i)
{
    // 异常判断
    if (i<0 || i>L.length)
    {
        cerr << "wrong erase position!" << endl;
        return false;
    }
    if (L.length == 0)
    {
        cerr << "List has no length" << endl;
        return false;
    }
    // 将位于删除位置之后的元素依次向前挪动一位
    for (int p = i + 1; p < L.length; ++p)
    {
        L.elem[p - 1] = L.elem[p];
    }
    // 线性长度-1
    L.length -= 1;
    return true;
    // 算法时间复杂度：O(n)
}


// 线性表的插入
bool InsertList(SqList &L, const ElemType &e, const int &i)
{
    // 判断线性表长度是否小于最大长度MAXSIZE
    if(L.length == MAXSIZE)
    {
        cerr << "can not insert!" << endl;
        return false;
    }
    if (i<0 || i>L.length)
    {
        cerr << "wrong insert position!" << endl;
        return false;
    }
    if (L.length > 0)
    {
        // 将位于插入位置之后的元素依次向后挪动一位
        for (int p = L.length - 1; p >= i; --p)
        {
            L.elem[p + 1] = L.elem[p];
        }
    }
    // 插入元素
    L.elem[i] = e;
    // 线性表长度+1
    L.length += 1;
    return true;
    // 算法时间复杂度：O(n)
}

void display_elem(SqList &L)
{
    for (int i = 0; i < L.length; i++)
    {
        cout << "elem" << i << " = " << L.elem[i] << endl;
    }
}

int main()
{
    SqList L;
    int a_e = 1;
    int a_i = 1;
    InitList(L);
    InsertList(L, a_e, a_i);

    cout << L.elem[1] << endl;
    cout << L.length << endl;
    int b_e = 2;
    int b_i = 2;
    InsertList(L, b_e, b_i);
    cout << L.elem[2] << endl;
    cout << L.length << endl;
    ClearList(L);
    display_elem(L);
    return 0;
}
