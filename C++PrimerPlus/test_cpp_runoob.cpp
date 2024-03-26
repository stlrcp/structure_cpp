/*
#include <iostream>
using namespace std;
struct ListNode{
    int data;
    ListNode *next;
} listNode;

ListNode* ListCreate(int n){
    ListNode *head, *p1, *end;
    head= new ListNode;
    end = head;
    for (int i = 0; i < n; i++){
        p1 = new ListNode;
        p1->data = i;
        end->next = p1;
        end = p1;
    }
    end->next = NULL;  // 链表的尾部 next 指向空
    return head;
}
ListNode* Find(ListNode *head, int n){
    ListNode *p = head;
    for (int i = 0; i < n && p != NULL; i++){
        p = p->next;
    }
    if (p==NULL){
        cout << "找不到/ToT/~~" << endl;
    }
    return p;
}
ListNode* ListInsert(ListNode *head, int n, int data){
    ListNode *p = head;
    ListNode *in = new ListNode;
    for (int i = 0; i < n && p != NULL; i++){
        p = p->next;
    }
    if (p!=NULL){
        in->data = data;
        in->next = p->next;
        p->next = in;
    }else{
        cout << "找不到/(ToT)~~" << endl;
    }
}
void print(ListNode *head){
    ListNode *temp;
    temp = head->next;
    while(temp!=NULL){
        cout << temp->data << " ";
        temp = temp->next;
    }
    cout << endl;
}
void ListChange(ListNode *head, int n , int temp){
    ListNode *p = head;
    for (int i = 0; i<n&&p!=NULL; i++){
        p = p->next;
    }
    if (p!=NULL){
        p->data = temp;
    } else{
        cout << "找不到节点~~" << endl;
    }
}
void ListDelete(ListNode *head, int n){
    ListNode *p = head, *p1 = head;
    for (int i = 0; i < n && p != NULL; i++){
        p1=p;
        p = p->next;
    }
    if (p!=NULL){
        p1->next = p->next;
        delete p;
    }else{
        cout << "找不到节点~~~~" << endl;
    }
}
int main(){
    int n = 6;
    ListNode *head = new ListNode;
    head = ListCreate(n);
    print(head);
    ListChange(head, 2, 999);
    print(head);
    ListInsert(head, 3, 1000);
    print(head);
    ListDelete(head, 3);
    print(head);
    return 0;
}
*/



/**
// 输出 "Hello, world"
#include <iostream>
using namespace std;
int mian(){
    cout << "Hello, World!";
    return 0;
}

// 标准输入输出
#include <iostream>
using namespace std;
int main(){
    int number;
    cout << "输入一个整数：";
    cin >> number;
    cout << "输出的数字为：" << number;
    return 0;
}

// 输出换行
#include <iostream>
using namespace std;
int main(){
    cout << "Runoob \n";
    cout << "Google \n";
    cout << "Taobao";
    return 0;
}

#include <iostream>
using namespace std;
int main(){
    cout << "Runoob" << endl;
    cout << "Google" << endl;
    cout << "Taobao";
    return 0;
}

// 实现两个数相加
#include <iostream>
using namespace std;
int main(){
    int firstNumber, secondNumber, sumOfTwoNumber;
    cout << "输入两个整数：";
    cin >> firstNumber >> secondNumber;
    sumOfTwoNumber = firstNumber + secondNumber;
    // 输出
    cout << firstNumber << " + " << secondNumber << " = " << sumOfTwoNumber;
    return 0;
}

// 不同变量
#include <iostream>
#include <string>
using namespace std;
int main(){
    int myNum = 5;
    float myFloatNum = 5.99;
    double myDoubleNum = 9.98;
    char myLetter = 'D';
    bool myBoolean = true;
    string myString = "Runoob";
    // 输出
    cout << "int: " << myNum << endl;
    cout << "float: " << myFloatNum << endl;
    cout << "double: " << myDoubleNum << endl;
    cout << "char: " << myLetter << "\n";
    cout << "bool: " << myBoolean << "\n";
    cout << "string: " << myString << "\n";
    return 0;
}

// 求商和余数
#include <iostream>
using namespace std;
int main(){
    int divisor, dividend, quotient, remainder;
    cout << "输入被除数：";
    cin >> dividend;
    cout << "输入除数：";
    cin >> divisor;
    quotient = dividend / divisor;
    remainder = dividend % divisor;

    cout << "商 = " << quotient << endl;
    cout << "余数 =" << remainder;
    return 0;
}

// 查看 int，float, double 和 char 变量大小
#include <iostream>
using namespace std;
int main(){
    cout << "char: " << sizeof(char) << " 字节" << endl;
    cout << "int: " << sizeof(int) << " 字节" << endl;
    cout << "float: " << sizeof(float) << " 字节" << endl;
    cout << "double: " << sizeof(double) << " 字节" << endl;
    return 0;
}

// 交换两个数
#include <iostream>
using namespace std;
int main(){
    int a = 5, b = 10, temp;
    cout << "交换之前：" << endl;
    cout << "a = " << a << ", b = " << b << endl;
    temp = a;
    a = b;
    b = temp;
    cout << "\n交换之后: " << endl;
    cout << "a = " << a << ", b = " << b << endl;
    return 0;
}

// 不使用临时变量
#include <iostream>
using namespace std;
int main(){
    int a = 5, b = 10;
    cout << "交换之前：" << endl;
    cout << "a = " << a << ", b = " << b << endl;
    a = a + b;
    b = a - b;
    a = a - b;
    cout << "\n 交换之后：" << endl;
    cout << "a = " << a << ", b=" << b << endl;
    return 0;
}

// 不使用临时变量，使用异或的方法
#include <iostream>
using namespace std;
int main(){
    int a = 9, b = 4;
    cout << "a = " << a << endl;
    cout << "b = " << b << endl;
    a = a ^ b;
    b = a ^ b;
    a = a ^ b;
    cout << "a : " << a << endl;
    cout << "b : " << b << endl;
    return 0;
}

// 使用指针
#include <iostream>
using namespace std;
void change(int *p, int *q){
    int ch;
    ch = *p;
    *p = *q;
    *q = ch;
}
int main(){
    int a, b;
    int *p1, *p2;
    cout << "Please input two numbers: " << endl;
    cin >> a >> b;
    p1 = &a;
    p2 = &b;
    change(p1, p2);
    cout << "After changing: " << endl;
    cout << *p1 << " " << *p2 << endl;
    return 0;
}

// 判断一个数为奇数偶数
#include <iostream>
using namespace std;
int main(){
    int n;
    cout << "输入一个整数：";
    cin >> n;
    if (n % 2 == 0)
        cout << n << " 为偶数。";
    else
        cout << n << " 为奇数。";
    return 0;
}

// 使用三元运算符
#include <iostream>
using namespace std;
int main(){
    int n;
    cout << "输入一个整数：";
    cin >> n;
    (n % 2 == 0) ? cout << n << " 为偶数。" : cout << n << " 为奇数。";
    return 0;
}
// 使用 与运算 判断
#include <iostream>
using namespace std;
int main(){
    int n = 1;
    cout << "输入一个整数：";
    cin >> n;
    if ((n & 1) == 0)
        cout << n << " 为偶数。";
    else
        cout << n << " 为奇数。";
    return 0;
}

// 判断元音/辅音
#include <iostream>
using namespace std;
int main(){
    char c;
    bool ischar;
    int isLowercaseVowel, isUppercaseVowel;
    cout << "输入一个字母：";
    cin >> c;
    ischar = ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'));
    if (ischar){
        // 小写字母元音
        isLowercaseVowel = (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u');
        // 大写字母元音
        isUppercaseVowel = (c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U');
        // if 语句判断
        if (isLowercaseVowel || isUppercaseVowel)
            cout << "是元音";
        else
            cout << "是辅音";
    } else {
        cout << "输入的不是字母。";
    }
    return 0;
}


#include <iostream>
using namespace std;
bool isVowel(char letter){
    // 将字母转换为小写
    letter = tolower(letter);
    // 判断字母是否为元音
    if (letter == 'a' || letter == 'e' || letter == 'i' ||letter =='o' || letter == 'u'){
        return true;
    } else {
        return false;
    }
}
int main() {
    char letter;
    cout << "请输入一个字母：";
    cin >> letter;
    if (isVowel(letter)){
        cout << letter << "是元音字母。" << endl;
    } else {
        cout << letter << "是辅音字母。" << endl;
    }
    return 0;
}


#include <iostream>
#include <cstring>
#include <ctype.h>
using namespace std;
int main(){
    char world;
    char c[10];
    bool isChar;
    cout << "请输入一个字母：" << endl;
    cin.get(c, 10);
    if (strlen(c) > 1){
        cout << "请输入一个字符而不是字符串" << endl;
        return 0;
    }
    world = c[0];
    isChar = ((world >= 'a' && world <= 'z') || (world >= 'A' && world <= 'Z'));
    if (isChar) {
        if (world=='a'||world=='e'|| world=='i'||world=='o'||world=='u'|| world=='A'||world=='E'||world=='I'||world=='O'||world=='U'){
            cout << world << "该字母为元音" << endl;
            return 0;
        }
        else
        {
            cout << world << "该字母为辅音" << endl;
            return 0;
        }
    }
    cout << "不是字母" << endl;
    return 0;
}

// 简化代码
#include <iostream>
#include <cstdio>
using namespace std;
bool marks[28];
int main(){
    marks['a' - 'a'] = marks['e' - 'a'] = marks['i' - 'a'] = marks['o' - 'a'] = marks['u' - 'a'] = true;
    printf("请输入字母: ");
    char ch = getchar();
    cout << "ch = " << ch << endl;
    if (ch <= 'Z')
        ch += 32;
    printf("%s", marks[ch - 'a'] ? "是元音" : "是辅音");
    return 0;
}

// 判断三个数中的最大数
#include <iostream>
using namespace std;
int main(){
    float n1, n2, n3;
    cout << "请输入三个数：";
    cin >> n1 >> n2 >> n3;
    if ( n1>=n2 && n1 >=n3){
        cout << "最大数为：" << n1;
    }
    if (n2 >= n1 && n2 >= n3){
        cout << "最大数为：" << n2;
    }
    if (n3 >= n1 && n3 >= n2) {
        cout << "最大数为：" << n3;
    }
    return 0;
}

// 使用 if else
#include <iostream>
using namespace std;
int main(){
    float n1, n2, n3;
    cout << "请输入三个数：";
    cin >> n1 >> n2 >> n3;
    if ((n1 >= n2) && (n1 >= n3))
        cout << "最大数为：" << n1;
    else if ((n2 >= n1 )&& (n2 >= n3))
        cout << "最大数为：" << n2;
    else
        cout << "最大数为：" << n3;
    return 0;
}

// 使用内嵌的 if else
#include <iostream>
using namespace std;
int main(){
    float n1, n2, n3;
    cout << "请输入三个数：";
    cin >> n1 >> n2 >> n3;
    if (n1 >= n2){
        if (n1 >= n3)
            cout << "最大数为：" << n1;
        else
            cout << "最大数为: " << n3;
    } else {
        if (n2 >= n3)
            cout << "最大数为：" << n2;
        else
            cout << "最大数为: " << n3;
    }
    return 0;
}

// 使用临时变量
#include <iostream>
using namespace std;
int main() {
    float n1, n2, n3, max;
    cout << "请输入三个数：";
    cin >> n1 >> n2 >> n3;
    if (n1 >= n2){
        max = n1;
    } else {
        max = n2;
    }
    if (n3 >= max){
        max = n3;
    }
    cout << "最大数为：" << max;
    return 0;
}

#include <iostream>
using namespace std;
int main(int argc, char *argv[]){
    float a, b, c, Max;
    cout << "请输入三个数，用空格隔开：" << endl;
    cin >> a >> b >> c;
    Max = a > b ? a : b;
    Max = Max > c ? Max : c;
    cout << "最大数为 " << Max;
    return 0;
}

// 只是用三元运算符，不用临时变量
#include <iostream>
using std::cin;
using std::cout;
using std::endl;
int main(){
    float num1, num2, num3;
    cout << "请输入三个数：" << endl;
    cin >> num1 >> num2 >> num3;
    cout << "三个数中的最大数为：";
    num1 > num2 ? (num1 > num3 ? cout << num1 : cout << num3) : (num2 > num3 ? cout << num2 : cout << num3);
    return 0;
}

// 求一元二次方程的根
#include <iostream>
#include <cmath>
using namespace std;
int main() {
    float a, b, c, x1, x2, discriminant, realPart, imaginaryPart;
    cout << "输入a, b 和 c: ";
    cin >> a >> b >> c;
    discriminant = b * b - 4 * a * c;
    if (discriminant > 0){
        x1 = (-b + sqrt(discriminant)) / (2 * a);
        x2 = (-b - sqrt(discriminant)) / (2 * a);
        cout << "Roots are real and different." << endl;
        cout << "x1 = " << x1 << endl;
        cout << "x2 = " << x2 << endl;
    }
    else if (discriminant ==0){
        cout << "实根相同：" << endl;
        x1 = (-b + sqrt(discriminant)) / (2 * a);
        cout << "x1 = x2 = " << x1 << endl;
    }
    else{
        realPart = -b / (2 * a);
        imaginaryPart = sqrt(-discriminant) / (2 * a);
        cout << "实根不同：" << endl;
        cout << "x1 = " << realPart << " + " << imaginaryPart << "i" << endl;
        cout << "x2 = " << realPart << " - " << imaginaryPart << "i" << endl;
    }
    return 0;
}


#include <iostream>
#include <math.h>
int main(){
    float a, b, c, dt, x1, x2;
    std::cout << "分别输入方程参数 a, b, c的值 " << std::endl;
    std::cin >> a >> b >> c;
    if (a ==0){
        std::cout << "一元二次方程 a 的值不能为0" << std::endl;
        return 0;
    }
    dt = pow(b, 2) - 4 * a * c;
    std::cout << "方程的根判别式 dt= " << dt << std::endl;
    if (dt ==0)
    {
        x1 = (-b + sqrt(pow(b, 2) - 4 * a * c)) / (2 * a);
        x2 = x1;
        std::cout << "方程有2个相等实数根" << std::endl;
        std::cout << "x1=x2 = " << x1 << std::endl;
    }
    if (dt > 0){
        x1 = (-b - sqrt(pow(b, 2) - 4 * a * c)) / (2 * a);
        x2 = (-b + sqrt(pow(b, 2) - 4 * a * c)) / (2 * a);
        std::cout << "方程有2个不等实数根" << std::endl;
        std::cout << "x1 = " << x1 << "\t"
                  << "x2 = " << x2 << std::endl;
    }
    if (dt < 0)
        std::cout << "方程无实数根, 函数图像与x轴不相交" << std::endl;
    return 0;
}

// 计算自然数之和
#include <iostream>
using namespace std;
int main(){
    int n, sum = 0;
    cout << "请输入一个正整数：";
    cin >> n;
    for (int i = 1; i <= n; ++i){
        sum += i;
    }
    cout << "Sum = " << sum;
    return 0;
}

// 判断闰年
#include <iostream>
using namespace std;
int main(){
    int year;
    cout << "输入年份：";
    cin >> year;
    if (year % 4 == 0){
        if (year %100 == 0){
            // 这里如果被 400 整除是闰年
            if (year % 400 == 0)
                cout << year << " 是闰年";
            else
                cout << year << " 不是闰年";
        }
        else
            cout << year << " 是闰年";
    }
    else
        cout << year << " 不是闰年";
    return 0;
}

// 求一个数的阶乘
#include <iostream>
using namespace std;
int main(){
    unsigned int n;
    unsigned long long factorial = 1;
    cout << "请输入一个整数：";
    cin >> n;
    for (int i = 1; i <= n; ++i){
        factorial *= i;
    }
    cout << n << " 的阶乘为："
         << " = " << factorial;
    return 0;
}


#include <iostream>
using namespace std;
long fact(int);
int main(){
    int n = 12;
    unsigned long long factorial = 1;
    factorial = fact(n);
    cout << n << "! = " << factorial << endl;
    return 0;
}
long fact(int ip){
    if (ip==1){
        return 1;
    } else {
        return ip * fact(ip - 1);
    }
}

// 创建各类三角形图案
#include <iostream>
using namespace std;
int main(){
    int rows;
    cout << "输入行数：";
    cin >> rows;
    for (int i = 1; i <= rows; ++i){
        for (int j = 1; j <= i; ++j){
            cout << "* ";
        }
        cout << "\n";
    }
    return 0;
}


#include <iostream>
using namespace std;
int main(){
    int rows;
    cout << "输入行数：";
    cin >> rows;
    for (int i = 1; i <= rows; ++i){
        // for (int j = 1; j <= i; ++j){
        for (int j = 1; j <= i; j++){
            cout << j << " ";
        }
        cout << "\n";
    }
    return 0;
}


#include <iostream>
using namespace std;
int main(){
    char input, alphabet = 'A';
    cout << "输入最后一个大写字母：";
    cin >> input;
    for (int i = 1; i <= (input - 'A' + 1); ++i){
        for (int j = 1; j <= i; ++j){
            cout << alphabet << " ";
        }
        ++alphabet;
        cout << endl;
    }
    return 0;
}


#include <iostream>
using namespace std;
int main(){
    int rows;
    cout << "输入行数：";
    cin >> rows;
    for (int i = rows; i >= 1; --i){
        for (int j = 1; j <= i; ++j){
            cout << "* ";
        }
        cout << endl;
    }
    return 0;
}


#include <iostream>
using namespace std;
int main(){
    int rows;
    cout << "输入行数：";
    cin >> rows;
    for (int i = rows; i >= 1; --i){
        for (int j = 1; j <= i; ++j){
            cout << j << " ";
        }
        cout << endl;
    }
    return 0;
}


#include <iostream>
using namespace std;
int main(){
    int space, rows;
    cout << "输入行数：";
    cin >> rows;
    for (int i = 1, k = 0; i <= rows; ++i, k=0){
        for (space = 1; space <= rows-i; ++space)
        {
            cout << "  ";
        }
        while (k != 2*i-1){
            cout << "* ";
            ++k;
        }
        cout << endl;
    }
    return 0;
}


#include <iostream>
using namespace std;
int main(){
    int rows, count = 0, count1 = 0, k = 0;
    cout << "输入行数：";
    cin >> rows;
    for (int i = 1; i <= rows; ++i){
        for (int space = 1; space <= rows - i; ++space)
        {
            cout << "  ";
            ++count;
        }
        while(k != 2*i-1){
            if (count <= rows-1){
                cout << i + k << " ";
                ++count;
            }else{
                ++count1;
                cout << i + k - 2 * count1 << " ";
            }
            ++k;
        }
        count1 = count = k = 0;
        cout << endl;
    }
    return 0;
}


#include <iostream>
using namespace std;
int main(){
    int rows;
    cout << "输入行数：";
    cin >> rows;
    for (int i = rows; i >= 1; --i){
        for (int space = 0; space < rows - i; ++space)
            cout << "  ";
        for (int j = i; j <= 2 * i - 1; ++j)
            cout << "* ";
        for (int j = 0; j < i - 1; ++j)
            cout << "* ";
        cout << endl;
    }
    return 0;
}


#include <iostream>
using namespace std;
int main(){
    int rows, coef = 1;
    cout << "Enter number of rows: ";
    cin >> rows;

    for (int i = 0; i < rows; i++){
        for (int space = 1; space <= rows - i; space++)
            cout << "  ";
        for (int j = 0; j <= i; j++){
            if (j==0 || i==0)
                coef = 1;
            else
                coef = coef * (i - j + 1) / j;
            cout << coef << "   ";
        }
        cout << endl;
    }
    return 0;
}


#include <iostream>
using namespace std;
int main(){
    int rows, number = 1;
    cout << "输入行数：";
    cin >> rows;
    for (int i = 1; i <= rows; i++){
        for (int j = 1; j <= i; ++j){
            cout << number << " ";
            ++number;
        }
        cout << endl;
    }
    return 0;
}


#include <iostream>
using namespace std;
int main(){
    int space, rows = 5;
    for (int i = 1; i <= rows; ++i){
        for (space = 1; space <= rows - i; ++space){
            cout << "  ";
        }
        int k = 2 * (i - 1) + 1;
        while(k>0){
            cout << "* ";
            --k;
        }
        cout << endl;
    }
    return 0;
}

// 求两数的最大公约数
#include <iostream>
using namespace std;
int main(){
    int n1, n2;
    cout << "输入两个整数: ";
    cin >> n1 >> n2;
    while (n1 != n2){
        if (n1 > n2)
            n1 -= n2;
        else
            n2 -= n1;
    }
    cout << "HCF = " << n1;
    return 0;
}


#include <iostream>
using namespace std;
 int main(){
     int n1, n2, hcf;
     cout << "输入两个整数：";
     cin >> n1 >> n2;
    // 如果 n2 大于 n1 交换两个变量
    if (n2 > n1){
        int temp = n2;
        n2 = n1;
        n1 = temp;
    }
    for (int i = 1; i <= n2; ++i){
        if (n1 % i == 0 && n2 % i == 0){
            hcf = i;
        }
    }
    cout << "HCF = " << hcf;
    return 0;
 }


#include <iostream>
#include <typeinfo>
#include <iomanip>
using namespace std;
int main(){
    int a, b;
    int MAX, MIN, i;
    cout << "输入两个数: ";
    cin >> a >> b;
    cout << "检查输入数据类型" << typeid(a).name() << setw(6) << typeid(b).name() << endl;
    // 建议输入数据非法时，做异常抛出 throw
    // 找到两个数中，最大的
    MAX = a ? b : a , b;
    // 确定除数，被除数
    if (a == MAX)
        MIN = b;
    else
        MIN = a;
    // 求最大公约数
    for (i = MIN; i > 1; i--){
        if ((MAX%i == 0)&&(MIN%i==0))
            break;
        else
            continue;
    }
    cout << "最大公约数：" << i << endl;
    cout << "最小公倍数：" << (a * b) / i;
    return 0;
}

// 辗转相除法
#include <iostream>
using namespace std;
int main(){
    int m, n;
    cout << "输入两个整数：";
    cin >> m >> n;
    cout << m << endl;
    cout << n << endl;
    int r = m % n;
    while (r!=0){
        m = n;
        n = r;
        r = m % n;
    }
    cout << "HCF = " << n << endl;
    return 0;
}

// 最小公倍数
#include <iostream>
using namespace std;
int main(){
    int n1, n2, max;
    cout << "输入两个数：";
    cin >> n1 >> n2;
    // 获取最大的数
    max = (n1 > n2) ? n1 : n2;
    do {
        if (max %n1 ==0 && max % n2 ==0){
            cout << "LCM = " << max;
            break;
        }
        else
            ++max;
    } while (true);
    return 0;
}


#include <iostream>
using namespace std;
int main(){
    int n1, n2, hcf, temp, lcm;
    cout << "输入两个数：";
    cin >> n1 >> n2;
    hcf = n1;
    temp = n2;
    while (hcf != temp){
        if (hcf > temp)
            hcf -= temp;
        else
            temp -= hcf;
    }
    lcm = (n1 * n2) / hcf;
    cout << "LCM = " << lcm;
    return 0;
}


// // 辗转相除法 - 递归
// int gcd(int x, int y){
//     return y ? gcd(y, x % y) : x;
// }

// 实现一个简单的计算器
#include <iostream>
using namespace std;
int main(){
    char op;
    float num1, num2;
    cout << "输入运算符：+、-、*、/ : ";
    cin >> op;
    cout << "输入两个数：";
    cin >> num1 >> num2;
    switch(op)
    {
        case '+':
            cout << num1 + num2;
            break;
        case '-':
            cout << num1 - num2;
            break;
        case '*':
            cout << num1 * num2;
            break;
        case '/':
            if (num2 == 0){
                cout << "error不能除以零";
                break;
            } else {
                cout << num1 / num2;
                break;
            }
        default:
            cout << "Error! 请输入正确运算符。";
            break;
    }
    return 0;
}


#include <iostream>
using namespace std;
double add(double num1, double num2){
    return num1 + num2;
}
double subtract(double num1, double num2){
    return num1 - num2;
}
double multiply(double num1, double num2){
    return num1 * num2;
}
double divide(double num1, double num2){
    if (num2 !=0){
        return num1 / num2;
    } else {
        cout << "错误：除数不能为零！" << endl;
        return 0;
    }
}
int main(){
    double num1, num2;
    char op;
    cout << "请输入第一个数字：";
    cin >> num1;
    cout << "请输入运算符（+、-、*、/）：";
    cin >> op;
    cout << "请输入第二个数字";
    cin >> num2;
    double result;
    switch (op){
        case '+':
            result = add(num1, num2);
            break;
        case '-':
            result = subtract(num1, num2);
            break;
        case '*':
            result = multiply(num1, num2);
            break;
        case '/':
            result = divide(num1, num2);
            break;
        default:
            cout << "错误：无效的运算符！" << endl;
            return 0;
        }
        cout << "结果：" << result << endl;
        return 0;
}


#include <iostream>
#include <map>
using namespace std;
float add(float num1, float num2) { return num1 + num2; }
float subtract(float num1, float num2) {
    return num1 - num2;
}
float multiply(float num1, float num2) { return num1 * num2; }
float divide(float num1, float num2) { return num1 / num2; }
int main(){
    map<char, float (*)(float, float)> fun;
    fun['+'] = add;
    fun['-'] = subtract;
    fun['*'] = multiply;
    fun['/'] = divide;
    char op;
    float num1, num2;
    cout << "请输入运算符：+、-、*、/ ：";
    cin >> op;
    cout << "请输入两个数：";
    cin >> num1 >> num2;
    try{
        if (fun.count(op) > 0)
            cout << "结果：" << fun[op](num1, num2) << endl;
        else
            cout << "错误！请输入正确的运算符。" << endl;
    } catch (const std::exception& e){
        cout << "错误！除数不能为零。" << endl;
    }
    return 0;
}


// 猴子吃桃问题
#include <iostream>
using namespace std;
int main(){
    int x, y, n;
    for (x = 1, n = 0; n < 9; y=(x+1)*2, x=y, n++)
        ;
    cout << "第一天共摘的桃子数量为 " << x << endl;
    return 0;
}


#include <iostream>
int main(){
    for (int x = 1, n = 9; n > 0; n--){
        std::cout << "第" << n << "天吃之前有" << (x + 1) * 2 << "个桃子" << std::endl;
        std::cout << "第" << n << "天吃完后有" << x << "个桃子" << std::endl;
        x = (x + 1) * 2;
    }
    return 0;
}


// 使用递归
#include <iostream>
using namespace std;
int returnPeach(int day, int peach){
    if (day > 1)
    {
        peach = peach * 2 + 2;
        day -= 1;
        return returnPeach(day, peach);
    }else{
        return peach;
    }
}
int main(void){
    int peach = returnPeach(10, 1);
    cout << "第一天共摘桃子的数量为：" << peach << endl;
    return 0;
}


#include <iostream>
using namespace std;
int num(int n){
    int i;
    if (n==1)
        i = 1;
    else
        i = 2 * (num(n - 1) + 1);
    return i;
}
int main(){
    cout << "桃子一共有：" << num(10) << "个" << endl;
    return 0;
}
**/

#include <iostream>
using namespace std;
int main(){
    int z = 0;
    for (int i = 1, n = 9; n > 0; n--, i=(i+1)*2, z=i){}
    cout << z << endl;
    for (int i = 1, n = 0; n > 0; n--, i=(i+1)*2){
        z = i;
    }
    cout << z << endl;
    return 0;
}
