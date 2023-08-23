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
*/

