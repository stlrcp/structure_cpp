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


//  可以使用静态成员变量清楚了解构造和析构函数的调用情况
#include <iostream>
using namespace std;
class Cpoint{
    public:
        static int value;
        static int num;
        Cpoint(int x, int y){
            xp = x;
            yp = y;
            value++;
            cout << "调用构造：" << value << endl;
        }

        ~Cpoint(){
            num++;
            cout << "调用析构：" << num << endl;
        }
    private:
        int xp, yp;
};

int Cpoint::value = 0;
int Cpoint::num = 0;
class CRect{
    public:
        CRect(int x1, int x2):mpt1(x1, x2), mpt2(x1, x2) {
            cout << "调用构造\n";
        }
        ~CRect() {
            cout << "调用析构\n";
        }
    private:
        Cpoint mpt1, mpt2;
};
int main()
{
        CRect p(10, 20);
        cout << "Hello, world! " << endl;
        return 0;
}


#include <iostream>
using namespace std;

class Cpoint{
    public:
        static int value;
        static int num;
        Cpoint(int x, int y) {
            xp = x;
            yp = y;
            value++;
            cout << "调用构造：" << value << endl;
            cout << this->xp << " " << this->yp << endl;
        }
        ~Cpoint() {
            num++;
            cout << "调用析构：" << num << endl;
            cout << this->xp << " 析构 " << this->yp << endl;
        }

    private:
        int xp, yp;
};

int Cpoint::value = 0;
int Cpoint::num = 0;
class CRect{
    public:
        CRect(int x1, int x2):mpt1(x1, x1),mpt2(x2, x2){
            cout << "调用构造\n";
        }
        ~CRect(){
            cout << "调用析构\n";
        }
    private:
        Cpoint mpt1, mpt2;
};
int main()
{
        CRect p(10, 20);
        cout << "hello, world!" << endl;
        return 0;
}
//  静态成员必须要有显示的初始化语句


// C++ 继承
// 面向对象程序设计中最重要的一个概念是继承。继承允许我们依据另一个类来定义一个类，这使得创建和维护一个应用程序变得更容易
// 这样做，也达到了重用代码功能和提高执行效率的效果
// 当创建一个类时，您不需要重新编写新的数据成员和成员函数，只需指定新建的类继承了一个已有的类的成员即可。这个已有的类称为基类，新建的类称为派生类。
// 继承代表了 is a 关系。例如，哺乳动物是动物，狗是哺乳动物，因此，狗是动物，等等。
// 基类
class Animal{
    // eat() 函数
    // sleep() 函数
};
// 派生类
class Dog : public Animal{
    // bark() 函数
}

// 基类 & 派生类
// 一个类可以派生自多个类，这意味着，它可以从多个基类继承数据和函数。定义一个派生类，我们使用一个类派生列表来指定基类。
// 类派生列表以一个或多个基类命名，形式如下：
class derived-class : access-specifier base-class 
// 其中，访问修饰符 access-specifier 是 public、protected 或 private 其中的一个，base-class 是之前定义过的某个类的名称。
// 如果未使用访问修饰符 class-specifier, 则默认为 private。
// 假设有一个基类 Shape, Rectangle 是它的派生类，如下所示：

#include <iostream>
using namespace std;
// 基类
class Shape
{
    public:
        void setWidth(int w)
        {
            width = w;
        }
        void setHeight(int h)
        {
            height = h;
        }
    protected:
        int width;
        int height;
};
// 派生类
class Rectangle : public Shape{
    public:
        int getArea(){
            return (width * height);
        }
};
int main(void)
{
        Rectangle Rect;

        Rect.setWidth(5);
        Rect.setHeight(7);
        // 输出对象的面积
        cout << "Total area: " << Rect.getArea() << endl;
        return 0;
}


// 访问控制和继承
// 派生类可以访问基类中所有的非私有成员。因此基类成员如果不想被派生类的成员函数访问，则应在基类中声明为 private。
// 一个派生类继承了所有的基类方法，但下列情况除外：
// - 基类的构造函数、析构函数和拷贝构造函数。
// - 基类的重载运算符。
// - 基类的友元函数。

// 继承类型
// 当一个类派生自基类，该基类可以被继承为 public、protected 或 private 几种类型。继承类型是通过上面讲解的访问修饰符 access-specifier 来指定的。
// 我们几乎不使用 protected 或 private 继承，通常使用 public 继承。当使用不同类型的继承时，遵循以下几个规则：
// - 公有继承（public）：当一个类派生自公有基类时，基类的公有成员也是派生类的公有成员，基类的保护成员也是派生类的保护成员，
// 基类的私有成员不能直接被派生类访问，但是可以通过调用基类的公有和保护成员来访问
// - 保护继承（protected）： 当一个类派生自保护基类时，基类的公有和保护成员将成为派生类的保护成员。
// - 私有继承（private）： 当一个类派生自私有基类时，基类的公有和保护成员将成为派生类的私有成员。

// 多继承
// 多继承即一个子类可以有多个父类，它继承了多个父类的特性。
// C++ 类可以从多个类继承成员，语法如下：
class <派生类名> : <继承方式1><基类名1>, <继承方式2><基类名2>, ...
{
    <派生类实体>
};
// 其中，访问修饰符继承方式是 public.protected 或 private 其中的一个，用来修饰每个基类，各个基类之间用逗号分隔，如上所示：

#include <iostream>
using namespace std;
// 基类
class Shape
{
    public:
        void setWidth(int w)
        {
            width = w;
        }
        void setHeight(int h)
        {
            height = h;
        }
    protected:
        int width;
        int height;
};
// 基类 PaintCost
class PaintCost
{
    public:
        int getCost(int area)
        {
            return area * 70;
        }
};
// 派生类
class Rectangle : public Shape, public PaintCost{
    public:
        int getArea()
        {
            return (width * height);
        }
};
int main(void)
{
        Rectangle Rect;
        int area;

        Rect.setWidth(5);
        Rect.setHeight(7);

        area = Rect.getArea();
        // 输出对象的面积
        cout << "Total area: " << Rect.getArea() << endl;
        // 输出总花费
        cout << "Total paint cost: $" << Rect.getCost(area) << endl;
        return 0;
}


// // 另外多继承(环状继承)，A->D, B->D, C->(A, B), 例如：
// class D{.......};
// class B: public D{.......};
// class A: public D{.......};
// class C: public B, public A{......};
// // 这个继承会使 D 创建两个对象，要解决上面问题就要用虚拟继承格式
// // 格式：class 类名：virtual 继承方式 父类名
// class D{........};
// class B: virtual public D{......};
// class A: virtual public D{......};
// class C: public B, public A{......};
// // 虚继承 - （在创建对象的时候会创建一个虚表）在创建父类对象的时候
// A: virtual public D
// B: virtual public D
#include <iostream>
using namespace std;
// 基类
class D{
    public:
        D() { cout << "D()" << endl; }
        ~D() { cout << "~D()" << endl; }
    protected:
        int d;
};

class B : virtual public D
{
    public:
        B() { cout << "B()" << endl; }
        ~B() { cout << "~B()" << endl; }
    protected:
        int b;
};

class A : virtual public D
{
    public:
        A() { cout << "A()" << endl; }
        ~A() { cout << "~A()" << endl; }
    protected:
        int a;
};

class C : public B, public A
{
    public:
        C() { cout << "C()" << endl; }
        ~C() { cout << "~C()" << endl; }
    protected:
        int c;
};

int main()
{
        cout << "Hello world!" << endl;
        C c;    // D, B, A, C
        cout << sizeof(c) << endl;
        return 0;
}


// 一个派生类继承了所有的基类方法，但下列情况除外
// - 基类的构造函数、析构函数和拷贝构造函数
// - 基类的重载运算符
// - 基类的友元函数
// 因此，我们不能够再子类的成员函数体中调用基类的构造函数来为成员变量进行初始化。例如下面这样是不可以的
#include <iostream>
using namespace std;
// 基类
class Shape
{
    public:
        Shape(int w, int h)
        {
            width = w;
            height = h;
        }
    protected:
        int width;
        int height;
};
// 派生类
class Rectangle: public Shape{
    public:
        Rectangle(int a, int b)
        {
            Shape(a, b);
        }
};

// 但是我们可以把基类的构造函数放在子类构造函数的初始化列表上，以此实现调用基类的构造函数来为子类从基类继承的成员变量初始化
#include <iostream>
using namespace std;
// 基类
class Shape
{
    public:
        Shape(int w, int h)
        {
            width = w;
            height = h;

            cout << "width = " << w << endl;
            cout << "height = " << h << endl;
        }

    protected:
        int width;
        int height;
};
// 派生类
class Rectangle : public Shape{
    public:
        Rectangle(int a, int b):Shape(a, b)
        {
            cout << " shape = " << a * b << endl;
        }
};
int main()
{
        cout << "Hello, world!" << endl;
        Rectangle rec(2,3);
        cout << sizeof(rec) << endl;
        return 0;
}


//  关于构造函数初始化列表的执行顺序进行补充
#include <iostream>
using namespace std;
class A{
    public:
        A(){
            cout << "call A()" << endl;
        }
};
class B:A{
    public:
        B(int val) : A(), value(val)
        {
            val = 10;   // 重新赋值
            cout << "call B()" << endl;
            cout << val << endl;
        }
    private:
        int value;
};
int main(){
        B b(10);
        return 0;
}    // 说明放在初始化列表的部分在构造函数之前执行
*/


