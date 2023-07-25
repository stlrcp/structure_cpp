/*
#include <iostream>

int main() {
    int algo = 2;
    char a;
    a = '1' + static_cast<char>(algo);
    std::cout << a << std::endl;
    return 0;
}




//  类的构造函数是类的一种特殊的成员函数，它会在每次创建类的新对象时执行
//  构造函数的名称与类的名称是完全相同的，并且不会返回任何类型，也不会返回 void。构造函数可用于为某些成员变量设置初始值。
#include <iostream>
using namespace std;
class Line
{
    public:
        void setLength(double len);
        double getLength(void);
        Line();    // 这是构造函数
    private:
        double length;
};
//  成员函数定义，包括构造函数
Line::Line(void)
{
        cout << "Object is being created" << endl;
}
void Line::setLength(double len)
{
        length = len;
}
double Line::getLength(void)
{
        return length;
}
//  程序的主函数
int main()
{
        Line line;
        // 设置长度
        line.setLength(6.0);
        cout << "Length of line : " << line.getLength() << endl;
        return 0;
}



// 类的析构函数
// 类的析构函数是类的一种特殊的成员函数，它会在每次删除所创建的对象时执行
// 析构函数额名称与类的名称是完全相同的，只是在前面加了个波浪号（~）作为前缀，它不会返回任何值，也不能带有任何参数。
// 析构函数有助于在跳出程序（比如关闭文件、释放内存等）前释放资源
#include <iostream>
using namespace std;

class Line
{
    public:
        void setLength(double len);
        double getLength(void);
        Line();     // 这是构造函数声明
        ~Line();     // 这是析构函数声明
    private:
        double length;
};
// 成员函数定义，包括构造函数
Line::Line(void)
{
        cout << "Object is being created" << endl;
}
Line::~Line(void)
{
        cout << "Object is being deleted" << endl;
}
void Line::setLength(double len)
{
        length = len;
}
double Line::getLength(void)
{
        return length;
}
// 程序的主函数
int main()
{
        Line line;
        // 设置长度
        line.setLength(6.0);
        cout << "Length of line : " << line.getLength() << endl;
        return 0;
}



#include <iostream>
#include <string>
using namespace std;

class Student
{
    public:
        string name;
        string number;
        char X;
        int year;
        Student(string, string, char, int);   // 构造函数声明
        void xianshi(void);    // 用于输出类成员的值
};
// 成员函数定义，包括构造函数
Student::Student(string N, string n, char x, int y){   // 利用构造函数给类的成员赋值
        name = N;
        number = n;
        X = x;
        year = y;
}
void Student::xianshi() {   //  输出成员的值
        cout << name << endl;
        cout << number << endl;
        cout << X << endl;
        cout << year << endl;
}
int main()          // 主函数
{
        cout << "输入姓名：";
        string N;
        cin >> N;
        cout << "输入学号：";
        string n;
        cin >> n;
        cout << "输入性别（M 或 W）：";
        char x;
        cin >> x;
        cout << "输入年龄：";
        int y;
        cin >> y;
        Student S(N, n, x, y);     // 定义对象并对构造函数赋值
        S.xianshi();            // 引用输出函数
        return 0;
}


// 初始化顺序最好要按照变量在类声明的顺序一致，否则会出现一些特殊情况
#include <iostream>
using namespace std;
class Student1 {
        public:
                int a;
                int b;
                void fprint() {
                        cout << "a = " << a << " b = " << b << endl;
                }
                Student1(int i):b(i),a(b){ }  // 异常顺序：发现 a 的值为 0，b 的值为2，说明初始化仅仅对 b 有效果，对 a 没有起到初始化作用
                // Student1(int i):a(i),b(a){ }   // 正常顺序：发现 a = b = 2 说明两个变量都是初始化了的

                Student1()              // 无参构造函数
                {
                        cout << "默认构造函数 Student1" << endl;
                }

                Student1(const Student1& t1)   // 拷贝构造函数
                {
                        cout << "拷贝构造函数 Student1 " << endl;
                        this->a = t1.a;
                }

                Student1& operator = (const Student1& t1)  // 赋值运算符
                {
                        cout << "赋值函数 Student1 " << endl;
                        this->a = t1.a;
                        return *this;
                }
};
class Student2
{
        public:
                Student1 test;
                Student2(Student1 &t1){
                        test = t1;
                }
                // Student2(Student1 &t1):test(t1){ }
};
int main()
{
                Student1 A(2);     // 进入默认构造函数
                // Student1 A;
                Student2 B(A);     // 进入拷贝构造函数
                A.fprint();        // 输出前面初始化的结果
}   // 由上面的例子可知，初始化列表的顺序要跟你在类声明额顺序要一致。否则像上面的这种特殊情况，有些变量就不会被初始化。



#include <iostream>
using namespace std;
class Student1 {
        public:
                int a = 0;
                int b = 0;

                void fprint() {
                        cout << "a = " << a << " b = " << b << endl;
                }

                Student1()
                {
                        cout << "无参构造函数Student1 " << endl;
                }

                Student1(int i):a(i),b(a)
                // Student1(int i):b(i),a(b)
                {
                        cout << "有参构造函数Student1 " << endl;
                }

                Student1(const Student1& t1)
                {
                        cout << "拷贝构造函数 Student1 " << endl;
                        this->a = t1.a;
                        this->b = t1.b;
                }

                Student1& operator = (const Student1& t1)   // 重载赋值运算符
                {
                        cout << "赋值函数 Student1 " << endl;
                        this->a = t1.a;
                        this->b = t1.b;
                        return *this;
                }
};
class Student2
{
        public:
                Student1 test;
                Student2(Student1& t1) {
                        t1.fprint();
                        cout << "D: ";
                        test = t1;
                }
                // Student2(Student1 &t1):test(t1){ }
};
int main()
{
                cout << "A: ";
                Student1 A;
                A.fprint();

                cout << "B: ";
                Student1 B(2);
                B.fprint();

                cout << "C: ";
                Student1 C(B);
                C.fprint();

                cout << "D: ";
                Student2 D(C);
                D.test.fprint();
}



// C++ 拷贝构造函数
// 拷贝构造函数是一种特殊的构造函数，他在创建对象时，是使用同一类中之前创建的对象来初始化新创建的对象。拷贝构造函数通常用于：
// - 通过使用另一个同类型的对象来初始化新创建的对象
// - 复制对象把它作为参数传递给函数
// - 复制对象，并从函数返回这个对象
#include <iostream>
using namespace std;
class Line
{
        public:
                int getLength(void);
                Line(int len);   // 简单的构造函数
                Line(const Line &obj);   // 拷贝构造函数
                ~Line();      // 析构函数
        private:
                int *ptr;
};
// 成员函数定义，包括构造函数
Line::Line(int len)
{
                cout << "调用构造函数" << endl;
                // 为指针分配内存
                ptr = new int;
                *ptr = len;
}

Line::Line(const Line &obj)
{
                cout << "调用拷贝构造函数并为指针 ptr 分配内存" << endl;
                ptr = new int;
                *ptr = *obj.ptr;   // 拷贝值
}

Line::~Line(void)
{
                cout << "释放内存" << endl;
                delete ptr;
}
int  Line::getLength(void)
{
                return *ptr;
}
void display(Line obj)
{
                cout << "Line 大小 ：" << obj.getLength() << endl;
}
// 程序的主函数
int main()
{
                Line line(10);
                display(line);
                return 0;
}    // 在这里，obj 是一个对象引用，该对象是用于初始化另一个对象的
*/


#include <iostream>
using namespace std;
class Line
{
        public:
                int getLength(void);
                Line(int len);     // 简单的构造函数
                Line(const Line &obj);    // 拷贝构造函数
                ~Line();
        private:
                int *ptr;
};
// 成员函数定义，包括构造函数
Line::Line(int len)
{
                cout << "调用构造函数" << endl;
                // 为指针分配内存
                ptr = new int;
                *ptr = len;
}
Line::Line(const Line &obj)
{
                cout << "调用拷贝构造函数并为指针 ptr 分配内存" << endl;
                ptr = new int;
                *ptr = *obj.ptr;   // 拷贝值
}
Line::~Line(void)
{
                cout << "释放内存" << endl;
                delete ptr;
}
int Line::getLength(void)
{
                return *ptr;
}
void display(Line obj)
{
                cout << "line 大小：" << obj.getLength() << endl;
}
// 程序的主函数
int main()
{
                Line line1(10);
                Line line2 = line1;   // 这里也调用了拷贝构造函数
                display(line1);
                display(line2);
                return 0;
}
