/*
// C++ 中 析构函数 和 class类的案列
#include <iostream>
using namespace std;
class T
{
    public:
    ~T(){cout << "析构函数被调用。";}   // 为了简洁，函数体可以直接写在定义的后面，此时函数为内联函数
};
int main()
{
    T *t = new T();   // 建立一个T类的指针对象t
    delete t;
    cin.get();
};



// 类成员的初始化顺序
#include <iostream>
using namespace std;
class A
{
    private:
        int n1;
        int n2;
    public:
        A():n2(0),n1(n2+2){}
        void Print(){
            cout << "n1: " << n1 << ", n2: " << n2 << endl;
        }
};
int main()
{
        A a;
        a.Print();  // 输出结果 n1:46, n2:0
        return 0;
}



// 构造函数主动调用
#include <iostream>
using namespace std;
class Person{
    private:
        int age;
    public:
        Person(int age) {
            this->age = age;
        }
        void Display() const {
            cout << "age = " << age << endl;
        }
};
void test(const Person &r) {
        r.Display();
}
int main()
{
        test(Person(20));    // 主动调用构造函数创建匿名对象
        return 0;
}



// 主动调用析构函数
#include <iostream>
using namespace std;
class Person{
    private:
        char *name;
    public:
        Person() {
            this->name = new char[10];
        }
        ~Person() {
            cout << "begin to delete" << endl;
            delete[] name;
        }
};
int main()
{
        Person obj;
        obj.~Person();
        return 0;
}   // 析构函数可以主动调用，但是C++规范一般不推荐这样做，主动调用析构函数可能会导致内存重复释放


// 声明类的静态成员变量
class Base
{
    public:
        Base(int val = 0):m_x(val){
            cout << "Base constructor" << endl;
        }
    public:
        static int m_x;
};


// 静态数据成员不能在类中初始化，一般在类外（在cpp文件中，而不是在头文件中）和main()函数之前初始化，缺省时初始化为0
class Base
{
    public:
        Base(int val = 0):m_x(val){
            cout << "Base constructor" << endl;
        }
    public:
        static int m_x;
};
int Base::m_x = 20;   // 只有在类中声明静态变量才加static关键字，如果是在类外初始化静态成员变量不能加static关键字，不然会报错


#include <iostream>
using namespace std;
class Base
{
    public:
        static int m_x;
        void Display() {
            cout << "m_x = " << m_x << endl;    // 类的非静态成员函数，直接访问静态成员变量
        }
};
int Base::m_x = 20;
int main()
{
        Base obj;
        Base *p = &obj;
        cout << "Base::m_x = " << Base::m_x << endl;
        cout << "Base::m_x = " << obj.m_x << endl;
        cout << "Base::m_x = " << p->m_x << endl;
        obj.Display();
        return 0;
}


// 类const static成员变量
// 类的static数据成员，和普通数据成员一样，不能在类的定义体中初始化，相反，static数据成员通常在定义时才初始化，但是，const static数据成员就可以在类的定义体中进行初始化
class Account{
    private:
        // const static常量在类体中进行初始化
        const static period = 20;
        double daily_tbl[period];
        double amount;
    public:
        static double rate(){
            return interestRate;
        }
        static void setRate(double newRate){
            interestRate = newRate;
        }
};


// 调用类的静态成员函数
#include <iostream>
class CExample
{
    public:
        static void Func(CExample *pobj);
    private:
        int m_age;
};
void CExample::Func(CExample *pobj)
{
        CExample *pThis = pobj;
        pThis->m_age = 100;  // 通过传进来的对象指针，在静态成员函数中间接访问非静态数据成员
        std::cout << pThis->m_age << std::endl;
}
int main()
{
        CExample obj;
        CExample::Func(&obj);
        return 0;
}


// C++ 类 & 对象
#include <iostream>
using namespace std;
class Box
{
    public:
        double length;
        double breadth;
        double height;
        // 成员函数声明
        double get(void);
        void set(double len, double bre, double hei);
};
// 成员函数定义
double Box::get(void)
{
        return length * breadth * height;
}
void Box::set(double len, double bre, double hei)
{
        length = len;
        breadth = bre;
        height = hei;
}
int main()
{
        Box Box1;    // 声明 Box1，类型为 Box
        Box Box2;    // 声明 Box2，类型为 Box
        Box Box3;    // 声明 Box3，类型为 Box
        double volume = 0.0;     // 用于存储体积
        // Box1 详述
        Box1.height = 5.0;
        Box1.length = 6.0;
        Box1.breadth = 7.0;
        // Box2 详述
        Box2.height = 10.0;
        Box2.length = 12.0;
        Box2.breadth = 13.0;
        // Box1 的体积
        volume = Box1.height * Box1.length * Box1.breadth;
        cout << "Box1 的体积：" << volume << endl;
        // Box2 的体积
        volume = Box2.height * Box2.length * Box2.breadth;
        cout << "Box2 的体积：" << volume << endl;
        // Box3 详述
        Box3.set(16.0, 8.0, 12.0);
        volume = Box3.get();
        cout << "Box3 的体积：" << volume << endl;
        return 0;
}


// 类的构造函数
#include <iostream>
using namespace std;
class Line
{
    public:
        void setLength(double len);
        double getLength(void);
        Line();    // 这里是构造函数
    private:
        double length;
};
// 成员函数定义，包括构造函数
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
// 程序的主函数
int main()
{
        Line line;
        // 设置长度
        line.setLength(6.0);
        cout << "Length of line : " << line.getLength() << endl;
        return 0;
}



// 带参数的构造函数
#include <iostream>
using namespace std;
class Line
{
    public:
        void setLength(double len);
        double getLength(void);
        Line(double len);   // 这是构造函数
    private:
        double length;
};
//  成员函数定义，包括构造函数
Line::Line(double len)
{
        cout << "Object is being created, length = " << len << endl;
        length = len;
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
        Line line(10.0);
        // 获取默认设置的长度
        cout << "Length of line : " << line.getLength() << endl;
        // 再次设置长度
        line.setLength(6.0);
        cout << "Length of line : " << line.getLength() << endl;
        return 0;
}



// 类的析构函数是类的一种特殊的成员函数，它会在每次删除所创建的对象时执行, 析构函数有助于在跳出程序（比如关闭文件、释放内存等）前释放资源
#include <iostream>
using namespace std;
class Line
{
    public:
        void setLength(double len);
        double getLength(void);
        Line();     // 这是构造函数声明
        ~Line();    // 这是析构函数声明
    private:
        double length;
};
// 成员函数定义，包括构造函数
Line::Line(void)
{
        cout << "Object is being created " << endl;
}
Line::~Line(void)
{
        cout << "Object is being deleted " << endl;
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
*/

