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
}        // 通过使用已有的同类型的对象来初始化新创建的对象


//   在C++中，下面三种对象需要调用拷贝构造函数！
// 1. 对象以值传递的方式传入函数参数
#include <iostream>
using namespace std;

class CExample
{
        private:
                int a;
        public:
                // 构造函数
                CExample(int b)
                {
                        a = b;
                        cout << "creat: " << a << endl;
                }
                // 拷贝构造
                CExample(const CExample& C)
                {
                        a = C.a;
                        cout << "copy : " << endl;
                }
                // 析构函数
                ~CExample()
                {
                        cout << "delete: " << a << endl;
                }
                void Show(){
                        cout << a << endl;
                }
};
// 全局函数，传入的是对象
void g_Fun(CExample C)
{
                cout << "test" << endl;
}
int main()
{
                CExample test(1);
                // 传入对象
                g_Fun(test);
                return 0;
}    // 调用 g_Fun()时，会产生以下几个重要步骤：
// (1) test对象传入形参时，会先会产生一个临时变量，就叫 C 吧。
// (2) 然后调用拷贝构造函数把 test 的值给 C，整个这两个步骤有点像：CExample C(test)
// (3) 等 g_Fun() 执行完后，析构掉 C 对象


 
// C++ primer p406: 拷贝构造函数是一种特殊的构造函数，具有单个形参，该形参（常用const修饰）是对该类类型的引用。
// 当定义一个新对象并用一个同类型的对象对它进行初始化时，将显示使用拷贝构造函数。当该类型的对象传递给函数或
// 从函数返回该类型的对象时，将隐式调用拷贝构造函数。

// C++ 支持两种初始化形式：
// 拷贝初始化 int a = 5; 和直接初始化 int a(5); 对于其他类型没有什么区别，对于类类型直接初始化直接调用实参匹配
// 的构造函数，拷贝初始化总是调用拷贝构造函数，也就是说：
// A x(2);   // 直接初始化，调用构造函数
// A y = x;   // 拷贝初始化，调用拷贝构造函数

// // 必须定义拷贝构造函数的情况：
// 只包含类类型成员或内置类型（但不是指针类型）成员的类，无须显式地定义拷贝构造函数也可以拷贝；
// 有的类有一个数据成员是指针，或者是有成员表示在构造函数中分配的其他资源，这两种情况下都必须定义拷贝构造函数。

// // 什么情况使用拷贝构造函数：
// 类的对象需要拷贝时，拷贝构造函数将会被调用。以下情况都会调用拷贝构造函数：
// （1） 一个对象以值传递的方式传入函数体
// （2） 一个对象以值传递的方式从函数返回
// （3） 一个对象需要通过另一个对象进行初始化


// 关于为什么当类成员中含有指针类型成员且需要对其分配内存时，一定要有总定义拷贝构造函数？
// 默认的拷贝构造函数实现的只能是浅拷贝，即直接将原对象的数据成员值依次复制给新对象中对应的数据成员，并没有为新对象另外分配内存资源。
// 这样，如果对象的数据成员是指针，两个指针对象实际上指向的是同一块内存空间。
// 在某些情况下，浅拷贝会带来数据安全方面的隐患。
// 当类的数据成员中有指针类型时，我们就必须定义一个特定的拷贝构造函数，该拷贝构造函数不仅可以实现原对象和新对象之间数据成员的拷贝，
// 而且可以为新的对象分配单独的内存资源，这就是深拷贝构造函数。
// 如何防止默认拷贝发生
// 声明一个私有的拷贝构造函数，这样因为拷贝构造函数是私有的，如果用户试图按值传递返回该类的对象，编译器会报告错误，从而可以避免按值传递或返回对象
// 总结：
// 当出现类的等号赋值时，会调用拷贝函数，在未定义显示拷贝构造函数的情况下，系统会调用默认的拷贝函数-- 即浅拷贝，它能够完成成员的一一复制。
// 当数据成员中没有指针时，浅拷贝是可行的。但当数据成员中有指针时，如果采用简单的浅拷贝，则两类中的两个指针将指向同一个地址，当对象快结束时，
// 会调用两次析构函数，而导致指针悬挂现象，所以，这时，必须采用深拷贝。
// 深拷贝与浅拷贝的区别就在于深拷贝会在堆内存中另外申请空间来储存数据，从而也就解决了指针悬挂的问题。
// 简而言之，当数据成员中有指针时，必须要用深拷贝。


#include <iostream>
using namespace std;

class A {
        public:
                int *pa;
                A(int x) {
                        cout << "调用构造函数初始化成员变量" << endl;
                        pa = new int;
                        *pa = x;
                }  // 构造
                A(const A& obj) {      // 拷贝构造
                        cout << "调用了拷贝构造函数，复制被调用的对象" << endl;
                        pa = new int;
                        *pa = *(obj.pa);   // 如果使用 pa = obj.pa 会出现错误，注意
                }
                ~A() {
                        cout << "调用析构函数释放内存" << endl;
                        delete pa;
                }
};
void transfer(A& obj) {
                cout << "pa 的地址：" << obj.pa << "\n pa的值: " << *obj.pa << endl;
}
A returnObj(A obj) {
                return obj;
}
int main() {
                A a(10);
                transfer(a);
                returnObj(a);
                return 0;
}



// 关于默认的浅拷贝构造函数，以下实例结合初始化列表的使用，自行定义浅拷贝构造函数：
#include <iostream>
using namespace std;
// *类定义*
class Box {
        private:
                int *length;
        public:
                Box(int *len);
                Box(const Box &obj);
                int GetBoxLen(void);
                ~Box();
};
// *类构造函数*
Box::Box(int *len) :length(len) {    // 将指针参数传入构造函数中，再利用初始化列表
                cout << "Object has created" << endl;
}
// ********类浅拷贝函数*********
Box::Box(const Box &obj):length(obj.length) {     // 直接传入同类型对象存储的内容
                cout << "Object has copied" << endl;    // 如果是指针，则代表指向同一片内存空间；
}

int Box::GetBoxLen(void) {
                return *length;
}
// 类析构函数
Box::~Box() {
                cout << "release" << endl;
                delete length;
}
int main() {
                int a = 10;
                Box box1(&a);
                Box box2 = box1;
                cout << "len of box1: " << box1.GetBoxLen() << endl;
                cout << "len of box2: " << box2.GetBoxLen() << endl;
                return 0;
}   //  执行出现错误，可以编译运行，但最后只输出一个类析构函数中的打印字符；可以理解为指向的内存空间被释放后，另一个析构函数释放该内存空间出现异常。
// 所以通过 new 运算符开辟内存空间，而不是简单的通过初始化列表来复制指针指向的内存空间地址，以定义深拷贝构造函数
*/


// 查看构造函数、拷贝构造函数、析构函数的调用情况
#include <iostream>
using namespace std;
class Line {
        public:
                int getLength(void);
                Line(int len, string name);   // 简单的构造函数
                Line(const Line &obj);    // 拷贝构造函数
                ~Line();     // 析构函数
                string name;
        private:
                int *ptr;
};
// 成员函数定义，包括构造函数
Line::Line(int len, string n)
{
        // 为指针分配内存
                ptr = new int;
                *ptr = len;
                name = n;
                cout << "调用析构函数 " << name << endl;
}
Line::Line(const Line &obj)
{
                cout << "调用拷贝构造函数 " << obj.name << endl;
                ptr = new int;
                *ptr = *obj.ptr;   // 拷贝值
                name = "~" + obj.name;
}
Line::~Line(void)
{
                cout << "释放内存" << name << endl;
                delete ptr;
}
int Line::getLength(void)
{
                return *ptr;
}
void display(Line obj)
{
                cout << "line 大小 : " << obj.getLength() << endl;
}
Line newline(double len)
{
                Line temp_line(len, "temp_line");
                return temp_line;
}
// 程序的主函数
int main() {
                Line line1(10, "direct_line1");    // 直接初始化，调用构造函数
                cout << endl;
                Line line2 = line1;    // 拷贝初始化，调用拷贝构造函数
                line2.name = "copy_line2";
                cout << endl;
                display(line1);    // 一个对象以值传递的方式传入函数体
                cout << endl;
                line1 = newline(8);   // 一个对象以值传递的方式从函数返回
                cout << line1.name << endl;
                line1.name = "modified_line1";
                cout << endl;
                display(line2);
                cout << endl;
                return 0;
}




