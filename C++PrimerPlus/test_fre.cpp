/*
// 类的友元函数是定义在类外部，但有权访问类的所有私有（private）成员和保护（protected）成员，尽管友元函数的原型有在类的定义中出现过，但是友元函数并不是成员函数。
// 友元可以是一个函数，该函数被称为友元函数；友元也可以是一个类，该类被称为友元类，在这种情况下，整个类及其所有成员都是友元。
// 如果要声明函数为一个类的友元，需要在类定义中该函数原型前使用关键字 friend，如下所示：
#include <iostream>
using namespace std;
class Box
{
    double width;
    public:
        friend void printWidth(Box box);
        void setWidth(double wid);
};
// 成员函数定义
void Box::setWidth(double wid)
{
        width = wid;
}
// 请注意：printWidth() 不是任何类的成员函数
void printWidth(Box box)
{
    // 因为 printWidth() 是 Box 的友元，它可以直接访问该类的任何成员
        cout << "Width of box : " << box.width << endl;
}
// 程序的主函数
int main()
{
        Box box;
        // 使用成员函数设置宽度
        box.setWidth(10.0);
        // 使用友元函数输出宽度
        printWidth(box);
        return 0;
}


// 友元函数的使用
// 因为友元函数没有 this 指针，则参数要有三种情况：
// 要访问非 static 成员时，需要对象做参数；
// 要访问 static 成员或全局变量时，则不需要对象做参数；
// 如果做参数的对象是全局对象，则不需要对象做参数
// 可以直接调用友元函数，不需要通过对象或指针
#include <iostream>
using namespace std;
class INTEGER
{
    friend void Print(const INTEGER &obj);    // 声明友元函数
};
void Print(const INTEGER& obj)
{
    // 函数体
    cout << &obj << endl;
}
int main() {
    INTEGER obj;
    Print(obj);   // 直接调用
}


#include <iostream>
using namespace std;
class Box
{
    double width;
    public:
        friend void printWidth(Box box);
        friend class BigBox;
        void setWidth(double wid);
};
class BigBox
{
    public:
        void Print(int width, Box &box)
        {
            // BigBox 是 Box 的友元类，它可以直接访问 Box 类的任何成员
            box.setWidth(width);
            cout << "Width of box : " << box.width << endl;
        }
};
// 成员函数定义
void Box::setWidth(double wid)
{
        width = wid;
}
// 请注意：printWidth() 不是任何类的成员函数
void printWidth(Box box)
{
    // 因为 printWidth() 是 Box 的友元，它可以直接访问该类的任何成员
        cout << "Width of box : " << box.width << endl;
}
// 程序的主函数
int main()
{
        Box box;
        BigBox big;
        // 使用成员函数设置宽度
        box.setWidth(10.0);
        // 使用友元函数输出宽度
        printWidth(box);
        // 使用友元类中的方法设置宽度
        big.Print(20, box);
        return 0;
}


// 内联函数
// C++ 内联函数是通常与类一起使用，如果一个函数是内敛的，那么在编译时，编译器会把该函数的代码副本放置在每个调用该函数的地方。
// 对内联函数进行任何修改，都需要重新编译函数的所有客户端，因为编译器需要重新更换一次所有的代码，否则将会继续使用旧额函数
#include <iostream>
using namespace std;
inline int Max(int x, int y)
{
    return (x > y) ? x : y;
}
// 程序的主函数
int main()
{
    cout << "Max (20, 10): " << Max(20, 10) << endl;
    cout << "Max (0, 200): " << Max(0, 200) << endl;
    cout << "Max (100, 1010): " << Max(100, 1010) << endl;
    return 0;
}
// 引入内联函数的目的是为了解决程序中函数调用的效率问题，这么说吧，程序在编译器编译的时候，编译器将程序中出现的内联函数的调用表达式用内联函数的
// 函数体进行替换，而对于其他的函数，都是在运行时才被替代。这其实就是个空间代价换时间的节省。所以内联函数一般都是1-5行的小函数，在使用内联函数时需注意：
// 1. 在内联函数内不允许使用循环语句和开关语句；
// 2. 内联函数的定义必须出现在内联函数第一次调用之前；
// 3. 类结构中所在的类说明内部定义的函数是内联函数。


// 在C++中，this指针是一个特殊的指针，它指向当前对象的实例。
// 在C++中，每一个对象都能通过 this 指针来访问自己的地址。
// 友元函数没有 this 指针，因为友元不是类的成员，只有成员函数才有 this 指针
#include <iostream>
class MyClass{
    private:
        int value;
    public:
        void setValue(int value) {
            this->value = value;
        }
        void printValue() {
            std::cout << "Value: " << this->value << std::endl;
        }
};
int main() {
    MyClass obj;
    obj.setValue(42);
    obj.printValue();
    return 0;
}


#include <iostream>
using namespace std;
class Box
{
    public:
        // 构造函数定义
        Box(double l=2.0, double b=2.0, double h=2.0)
        {
            cout << "调用构造函数：" << endl;
            length = l;
            breadth = b;
            height = h;
        }
        double Volume()
        {
            return length * breadth * height;
        }
        int compare(Box box)
        {
            return this->Volume() > box.Volume();
        }
    private:
        double length; 
        double breadth;
        double height;
};
int main()
{
    Box Box1(3.3, 1.2, 1.5);
    Box Box2(8.5, 6.0, 2.0);
    if (Box1.compare(Box2))
    {
        cout << "Box2 的体积比 Box1 小 " << endl;
    } else {
        cout << "Box2 的体积大于或等于 Box1 " << endl;
    }
    return 0;
}


#include <iostream>
using namespace std;
class Box{
    public:
        Box(){;}
        ~Box(){;}
        Box* get_address()  // 得到this的地址
        {
            return this;
        }
};
int main(){
    Box box1;
    Box box2;
    // Box* 定义指针 p 接受对象 box 的get_address() 成员函数的返回值，并打印
    Box* p = box1.get_address();
    cout << p << endl;

    p = box2.get_address();
    cout << p << endl;
    return 0;
}
// this 指针的类型可理解为 Box*
// 此时得到两个地址分别为 box1 和 box2 对象的地址

// 内联函数
// C++ 内联函数是通常与类一起使用，如果一个函数是内敛的，那么在编译时，编译器会把该函数的代码副本放置在每个调用该函数的地方。
// 对内联函数进行任何修改，都需要重新编译函数的所有客户端，因为编译器需要重新更换一次所有的代码，否则将会继续使用旧额函数
#include <iostream>
using namespace std;
inline int Max(int x, int y)
{
    return (x > y) ? x : y;
}
// 程序的主函数
int main()
{
    cout << "Max (20, 10): " << Max(20, 10) << endl;
    cout << "Max (0, 200): " << Max(0, 200) << endl;
    cout << "Max (100, 1010): " << Max(100, 1010) << endl;
    return 0;
}
// 引入内联函数的目的是为了解决程序中函数调用的效率问题，这么说吧，程序在编译器编译的时候，编译器将程序中出现的内联函数的调用表达式用内联函数的
// 函数体进行替换，而对于其他的函数，都是在运行时才被替代。这其实就是个空间代价换时间的节省。所以内联函数一般都是1-5行的小函数，在使用内联函数时需注意：
// 1. 在内联函数内不允许使用循环语句和开关语句；
// 2. 内联函数的定义必须出现在内联函数第一次调用之前；
// 3. 类结构中所在的类说明内部定义的函数是内联函数。


// 在C++中，this指针是一个特殊的指针，它指向当前对象的实例。
// 在C++中，每一个对象都能通过 this 指针来访问自己的地址。
// 友元函数没有 this 指针，因为友元不是类的成员，只有成员函数才有 this 指针
#include <iostream>
class MyClass{
    private:
        int value;
    public:
        void setValue(int value) {
            this->value = value;
        }
        void printValue() {
            std::cout << "Value: " << this->value << std::endl;
        }
};
int main() {
    MyClass obj;
    obj.setValue(42);
    obj.printValue();
    return 0;
}


#include <iostream>
using namespace std;
class Box
{
    public:
        // 构造函数定义
        Box(double l=2.0, double b=2.0, double h=2.0)
        {
            cout << "调用构造函数：" << endl;
            length = l;
            breadth = b;
            height = h;
        }
        double Volume()
        {
            return length * breadth * height;
        }
        int compare(Box box)
        {
            return this->Volume() > box.Volume();
        }
    private:
        double length; 
        double breadth;
        double height;
};
int main()
{
    Box Box1(3.3, 1.2, 1.5);
    Box Box2(8.5, 6.0, 2.0);
    if (Box1.compare(Box2))
    {
        cout << "Box2 的体积比 Box1 小 " << endl;
    } else {
        cout << "Box2 的体积大于或等于 Box1 " << endl;
    }
    return 0;
}


#include <iostream>
using namespace std;
class Box{
    public:
        Box(){;}
        ~Box(){;}
        Box* get_address()  // 得到this的地址
        {
            return this;
        }
};
int main(){
    Box box1;
    Box box2;
    // Box* 定义指针 p 接受对象 box 的get_address() 成员函数的返回值，并打印
    Box* p = box1.get_address();
    cout << p << endl;

    p = box2.get_address();
    cout << p << endl;
    return 0;
}
// this 指针的类型可理解为 Box*
// 此时得到两个地址分别为 box1 和 box2 对象的地址


// 指向 类的 指针
// 一个指向 C++ 类的指针与指向结构的指针类似，访问执行类的指针的成员，需要使用成员访问运算符 -> , 就像访问指向结构的指针一样
// 与所有的指针一样，您必须在使用指针之前，对指针进行初始化
#include <iostream>
using namespace std;
class Box {
    public:
        // 构建函数定义
        Box(double l=2.0, double b=2.0, double h=2.0)
        {
            cout << "Constructor called." << endl;
            length = l;
            breadth = b;
            height = h;
        }
        double Volume()
        {
            return length * breadth * height;
        }
    private:
        double length;    // Length of a box
        double breadth;    // Breadth of a box
        double height;    // Height of a box
};

int main(void)
{
        Box Box1(3.3, 1.2, 1.5);    // Declare box1
        Box Box2(8.5, 6.0, 2.0);     // Declare box2
        Box *ptrBox;        // Declare pointer to a class

        // 保存第一个对象的地址
        ptrBox = &Box1;

        // 现在尝试使用成员访问运算符来访问成员
        cout << "Volume of  Box1: " << ptrBox->Volume() << endl;

        // 保存第二个对象的地址
        ptrBox = &Box2;

        // 现在尝试使用成员访问运算符来访问成员
        cout << "Volume of Box2: " << ptrBox->Volume() << endl;
        return 0;
}


// C++ 类的静态成员
// 我们可以使用 static 关键字来把类成员定义为静态的。当我们声明类的成员为静态时，这意味着无论创建多少个类的对象，静态成员都只有一个副本。
// 静态成员在类的所有对象中是共享的。如果不存在其他的初始化语句，在创建第一个对象时，所有的静态数据都会被初始化为零。
// 我们不能把静态成员的初始化放置在类的定义中，但是可以在类的外部通过使用范围解析运算符::来重新声明静态变量从而对它进行初始化
#include <iostream>
using namespace std;
class Box
{
    public:
        static int objectCount;
        // 构造函数定义
        Box(double l=2.0, double b=2.0, double h=2.0)
        {
            cout << "Constructor called. " << endl;
            length = l;
            breadth = b;
            height = h;
            // 每次创建对象时增加 1
            objectCount++;
        }
        double Volume()
        {
            return length * breadth * height;
        }
    private:
        double length;   // 长度
        double breadth;     // 宽度
        double height;    // 高度
};
// 初始化类 Box 的静态成员
int Box::objectCount = 0;
int main(void)
{
        Box Box1(3.3, 1.2, 1.5);    // 声明 Box1
        Box Box2(8.5, 6.0, 2.0);    // 声明 Box2

        // 输出对象的总数
        cout << "Total objects: " << Box::objectCount << endl;
        return 0;
}


// 静态成员函数
// 如果把函数成员声明为静态的，就可以把函数与类的任何特定对象独立开来。静态成员函数即时在类对象不存在的情况下也能被调用，
// 静态函数只要使用类名加范围解析运算符 ：：就可以访问
// 静态成员函数只能访问静态成员数据、其他静态成员函数和类外部的其他函数
// 静态成员函数有一个类范围，他们不能访问类的 this 指针。您可以使用静态成员函数来判断类的某些对象是否已被创建。
// 静态成员函数与普通成员函数的区别：
// 静态成员函数没有 this 指针，只能访问静态成员（包括静态成员变量和静态成员函数）
// 普通成员函数有 this 指针，可以访问类中的任意成员；而静态成员函数没有 this 指针。
#include <iostream>
using namespace std;

class Box
{
    public:
        static int objectCount;
        // 构造函数定义
        Box(double l =2.0, double b=2.0, double h=2.0)
        {
            cout << "Constructor called." << endl;
            length = l;
            breadth = b;
            height = h;
            // 每次创建对象时增加 1
            objectCount++;
        }
        double Volume()
        {
            return length * breadth * height;
        }
        static int getCount()
        {
            return objectCount;
        }
    private:
        double length;     // 长度
        double breadth;    // 宽度
        double height;     // 高度
};
// 初始化类 Box 的静态成员
int Box::objectCount = 0;

int main(void)
{
    // 在创建对象之前输出对象的总数
        cout << "Inital Stage Count: " << Box::getCount() << endl;

        Box Box1(3.3, 1.2, 1.5);   // 声明 Box1
        cout << "Final Stage Count: " << Box::getCount() << endl;
        Box Box2(8.5, 6.0, 2.0);    // 声明 Box2

        // 在创建对象之后输出对象的总数
        cout << "Final Stage Count: " << Box::getCount() << endl;
        return 0;
}


// 静态成员变量在类中仅仅是声明，没有定义，所以要在类的外面定义，实际上是给静态成员变量分配内存，如果不加定义就会报错，
// 初始化是赋一个初始值，而定义是分配内存
#include <iostream>
using namespace std;
class Box
{
    public:
        static int objectCount;
        // 构造函数定义
        Box(double l = 2.0, double b = 2.0, double h=2.0)
        {
            cout << "Constructor called." << endl;
            length = l;
            breadth = b;
            height = h;
            // 每次创建对象时增加 1
            objectCount++;
        }
        double Volume()
        {
            return length * breadth * height;
        }
    private:
        double length;   // 长度
        double breadth;   // 宽度
        double height;    // 高度
};
// 初始化类 Box 的静态成员，其实是定义并初始化的过程
// int Box::objectCount = 0;
// 也可这样 定义却不初始化
int Box::objectCount;
int main(void)
{
        Box Box1(3.3, 1.2, 1.5);   // 声明 box1
        Box Box2(8.5, 6.9, 2.0);    // 声明 box2
        // 输出对象的总数
        cout << "Total object: " << Box::objectCount << endl;
        return 0;
}
*/


