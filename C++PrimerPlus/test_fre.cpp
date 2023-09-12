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


// 派生类在继承基类的成员变量时，会单独开辟一块内存保存基类的成员变量，因此派生类自己的成员变量即使和基类的成员变量重名，但是也不会引起冲突。
#include <iostream>
using namespace std;
// 基类
class A{
    public:
        A() { n = 0; }
        ~A() { cout << "析构函数" << endl; }
        int getA() { return n; }
        void setA(int t) { n = t; }
    private:
        int n;
};
// 派生类
class B : public A {
    public:
        B() { n = 0; }
        ~B() { cout << "~B()" << endl; }
        int getB() { return n; }
        void setB(int t) { n = t; }
    private:
        int n;
};
int main(int argc, char* argv[])
{
        B b;
        b.setA(10); // 设置基类的成员变量n

        cout << "A::n = " << b.getA() << endl;
        cout << "B::n = " << b.getB() << endl;

        b.setB(9);   // 设置派生类的成员变量n
        cout << "A::n = " << b.getA() << endl;
        cout << "B::n = " << b.getB() << endl;
        return 0;
}


// 构造函数调用顺序：基类 > 成员类 > 派生类；
// 多继承派生类：基类构造顺序 依据 基类继承顺序调用
// 类成员：依据类成员对象 定义顺序 调用成员类构造函数
#include <iostream>
using namespace std;
class Shape{    // 基类  Shape
    public:
        Shape() {
            cout << "Shape" << endl;
        }
        ~Shape() {
            cout << "~Shape()" << endl;
        }
};

class PaintCost{   // 基类 PaintCost
    public:
        PaintCost() {
            cout << "PaintCost" << endl;
        }
        ~PaintCost() {
            cout << "~PaintCost" << endl;
        }
};
// 派生类
class Rectangle : public Shape, public PaintCost  // 基类构造顺序 依照 继承顺序
{
    public:
        Rectangle(): b(), a(), Shape(), PaintCost(){
            cout << "Rectangle" << endl;
        }
        ~Rectangle() {
            cout << "~Rectangle" << endl;
        }
        PaintCost b;    // 类成员变量构造顺序 依据 变量定义顺序
        Shape a;
};
int main(void)
{
        Rectangle Rect;
        return 0;
}


// C++ 重载运算符和重载函数
// C++ 允许在同一作用域中的某个函数和运算符指定多个定义，分别称为函数重载和运算符重载
// 重载声明是指一个与之前已经在该作用域内声明过的函数或方法具有相同名称的声明，但是他们的参数列表和定义（实现）不相同。
// 当您调用一个重载函数或重载运算符时，编译器通过把您所使用的参数类型与定义中的参数类型进行比较，决定选用最合适的定义。选用最合适的重载函数或重载运算符的过程，称为重载决策。
// C++ 中的函数重载
// 在同一个作用域内，可以声明几个功能类似的同名函数，但是这些同名函数的形式参数（指参数的个数、类型或者顺序）必须不同。您不能仅通过返回类型的不同来重载函数。
#include <iostream>
using namespace std;
class printData{
    public:
        void print(int i){
            cout << "整数为：" << i << endl;
        }
        void print(double f){
            cout << "浮点数为：" << f << endl;
        }
        void print(char c[]){
            cout << "字符串为：" << c << endl;
        }
};
int main(void)
{
        printData pd;
        // 输出整数
        pd.print(5);
        // 输出浮点数
        pd.print(500.265);
        // 输出字符串
        char c[] = "Hello C++";
        pd.print(c);
        return 0;
}


// C++ 中的运算符重载
// 您可以重定义或重载大部分 C++ 内置的运算符。这样，您就能使用自定义类型的运算符。
// 重载的运算符是带有特殊名称的函数，函数名是由关键字 operator 和其后要重载的运算符符号构成的，与其他函数一样，重载运算符有一个返回类型和一个参数列表。
// Box operator+(const Box&);
// 声明加法运算符用于把两个 Box 对象相加，返回最终的 Box 对象。大多数的重载运算符可被定义为普通的非成员函数或者被定义为类成员函数。
// 如果我们定义上面的函数为类的非成员函数，那么我们需要为每次操作传递两个参数，如下所示：
// Box operator+(const Box&, const Box&);
// 下面实例使用成员函数演示了运算符重载的概念。在这里，对象作为参数进行传递，对象的属性使用 this 运算符进行访问。
#include <iostream>
using namespace std;
class Box{
    public:
        double getVolume(void){
            return length * breadth * height;
        }
        void setLength(double len){
            length = len;
        }
        void setBreadth(double bre){
            breadth = bre;
        }
        void setHeight(double hei){
            height = hei;
        }
        // 重载 + 运算符，用于把两个 Box 对象相加
        Box operator+(const Box& b){
            Box box;
            box.length = this->length + b.length;
            box.breadth = this->breadth + b.breadth;
            box.height = this->height + b.height;
            return box;
        }
    private:
        double length;
        double breadth;
        double height;
};
// 程序的主函数
int main(){
        Box Box1;
        Box Box2;
        Box Box3;
        double volume = 0.0;   // 把体积存储在该变量中
        // Box1 详述
        Box1.setLength(6.0);
        Box1.setBreadth(7.0);
        Box1.setHeight(5.0);
        // Box2 详述
        Box2.setLength(12.0);
        Box2.setBreadth(13.0);
        Box2.setHeight(10.0);
        // Box1 的体积
        volume = Box1.getVolume();
        cout << "Volume of Box1 : " << volume << endl;
        // Box2 的体积
        volume = Box2.getVolume();
        cout << "Volume of Box2 : " << volume << endl;
        // 把两个对象相加，得到 Box3
        Box3 = Box1 + Box2;
        // Box3 的体积
        volume = Box3.getVolume();
        cout << "Volume of Box3 : " << volume << endl;
        return 0;
}


// 注意：
// 1. 运算符重载不可以改变语法结构与。
// 2. 运算符重载不可以改变操作数的个数。
// 3. 运算符重载不可以改变优先级。
// 4. 运算符重载不可以改变结合性。

// 类重载、覆盖、重定义之间的区别：
// 重载指的是函数具有不同的参数列表，而函数名相同的函数。重载要求参数列表必须不同，比如参数的类型不同、参数的个数不同、参数的顺序不同。
// 如果仅仅是函数的返回值不同是没办法重载的，因为重载要求参数列表必须不同。（发生在同一个类里）
// 覆盖是存在类中，子类重写从基类继承过来的函数。被重写的寒素不能是 static 的。必须是 virtual 的。但是函数名、返回值、参数列表都必须和基类相同（发生在基类和子类）
// 重定义也叫做隐藏，子类重新定义父类中有相同名称的非虚函数（参数列表可以不同）。（发生在基类和子类）

// this 指针的作用
// this 指针是一个隐含于每一个非静态成员函数中的特殊指针。它指向正在被该成员函数操作的那个对象。
// 当对一个对象调用成员函数时，编译器先将对象的地址赋给 this 指针，然后调用成员函数，每次成员函数存取数据成员时由隐含使用 this 指针
// 运算符重载的同时也可以发生函数重载
#include <iostream>
using namespace std;
// 加号运算符重载
class xiMeng{
    public:
        int M_A;
        int M_B;
        // 通过成员函数运算符重载
        // xiMeng operator+(xiMeng & p)
        // {
        //     xiMeng temp;
        //     temp.M_A = this->M_A + p.M_A;
        //     temp.M_B = this->M_B + p.M_B;
        //     return temp;
        // }
};
// 通过全局函数运算符重载
xiMeng operator+ (xiMeng & p1, xiMeng & p2)
{
        xiMeng temp;
        temp.M_A = p1.M_A + p2.M_A;
        temp.M_B = p1.M_B + p2.M_B;
        return temp;
}
// 运算符重载也可以发生函数重载
xiMeng operator+(xiMeng& p, int num)
{
        xiMeng temp;
        temp.M_A = p.M_A + num;
        temp.M_B = p.M_B + num;
        return temp;
}
void xiMengTest(){
        xiMeng p1;
        p1.M_A = 15;
        p1.M_B = 25;

        xiMeng p2;
        p2.M_A = 10;
        p2.M_B = 30;
        // 通过全局函数运算符重载
        xiMeng p3 = p1 + p2;
        cout << "p3.M_A = " << p3.M_A << endl;
        cout << "p3.M_B = " << p3.M_B << endl;

        // 运算符重载也可以发生函数重载
        xiMeng p4 = p1 + 100;
        cout << "p4.M_A = " << p4.M_A << endl;
        cout << "p4.M_B = " << p4.M_B << endl;
}
int main(){
        xiMengTest();
        return 0;
}


// C++ 一元运算符重载
// 一元运算符只对一个操作数进行操作，下面是一元运算符的实例：
// - 递增运算符（++） 和递减运算符（--）
// - 一元减运算符，即负号（-）
// - 逻辑非运算符（！）
// 一元运算符通常出现在它们所操作的对象的左边，比如 !obj, -obj 和 ++obj, 但有时它们也可以做为后缀，比如 obj++ 或 obj--。
// 实例
#include <iostream>
using namespace std;
class Distance{
    private:
        int feet;   // 0 到 无穷
        int inches;    // 0 到 12
    public:
        // 所需的构造函数
        Distance(){
            feet = 0;
            inches = 0;
        }
        Distance(int f, int i){
            feet = f;
            inches = i;
        }
        // 显示距离的方法
        void displayDistance(){
            cout << "F: " << feet << " I: " << inches << endl;
        }
        // 重载负运算符（-）
        Distance operator- ()
        {
            feet = -feet;
            inches = -inches;
            return Distance(feet, inches);
        }
};
int main()
{
        Distance D1(11, 10), D2(-5, 11);
        -D1;   // 取相反数
        D1.displayDistance();   // 距离 D1
        -D2;   // 取相反数
        D2.displayDistance();   // 距离 D2
        return 0;
}


// 重载单目运算符++ （或 - ）作为前缀和后缀：
// 前缀和后缀重载的语法格式是不同的
#include <iostream>
using namespace std;
class Complex{
    private:
        double i;
        double j;
    public:
        Complex(int = 0, int = 0);
        void display();
        Complex operator++();  // 前缀自增
        Complex operator++(int);  // 后缀自增，参数需要加 int
};
Complex::Complex(int a, int b){
        i = a;
        j = b;
}
void Complex::display() {
        cout << i << " + " << j << "i" << endl;
}
Complex Complex::operator ++() {
        ++i;
        ++j;
        return *this;
}
Complex Complex::operator++(int){
        Complex temp = *this;
        ++*this;
        return temp;
}
int main()
{
        Complex comnum1(2, 2), comnum2, comnum3;
        cout << "自增计算前：" << endl;
        cout << "comnum1: ";
        comnum1.display();
        cout << "comnum2: ";
        comnum2.display();
        cout << "comnum3: ";
        comnum3.display();
        cout << endl;

        cout << "前缀自增计算后：" << endl;
        comnum2 = ++comnum1;
        cout << "comnum1: ";
        comnum1.display();
        cout << "comnum2: ";
        comnum2.display();
        cout << endl;

        cout << "后缀自增计算后：" << endl;
        comnum3 = comnum1++;
        cout << "comnum1: ";
        comnum1.display();
        cout << "comnum3: ";
        comnum3.display();
        return 0;
}


// 在前缀递增时，若想要实现 ++(++a) 这种连续自加，就要返回其对象的引用，这样才能保证操作的是同一块内存空间，否则就只是单纯的赋值操作，原来的对象并未被修改
#include <iostream>
using namespace std;
class Complex{
    private:
        double i;
        double j;
    public:
        Complex(int = 0, int = 0);
        void display();
        Complex &operator++();  // 前缀自增
        Complex operator++(int);    // 后缀自增，参数需要加 int
};
Complex::Complex(int a, int b){
        i = a;
        j = b;
}
void Complex::display(){
        cout << "i = " << i << "\t j = " << j << endl;
}
Complex& Complex::operator++(){
        ++i;
        ++j;
        return *this;
}
Complex Complex::operator ++(int){
        Complex temp = *this;
        ++*this;
        return temp;
}
int main()
{
        Complex comnum1(2, 2), comnum2, comnum3;
        cout << "自增计算前： " << endl;
        cout << "comnum1: ";
        comnum1.display();
        cout << "comnum2: ";
        comnum2.display();
        cout << "comnum3: ";
        comnum3.display();
        cout << endl;

        cout << "前缀自增计算后：" << endl;
        comnum2 = ++comnum1;
        cout << "comnum1: ";
        comnum1.display();
        cout << "comnum2: ";
        comnum2.display();
        cout << endl;

        cout << "后缀自增计算后：" << endl;
        comnum3 = comnum1++;
        cout << "comnum1: ";
        comnum1.display();
        cout << "comnum3: ";
        comnum3.display();

        cout << "前缀递增加引用是为了连续自加++(++a)" << endl;
        Complex comcum4;
        ++(++comcum4);
        cout << "comcum4: ";
        comcum4.display();
        return 0;
}


// C++ 二元运算符重载
// 二元运算符需要两个参数，下面是二元运算符的实例。我们平常使用的加运算符（+）、减运算符（-）、乘运算符（*）和除运算符（/）都属于二元运算符。
// 实例
#include <iostream>
using namespace std;

class Box{
    double length;    // 长度
    double breadth;
    double height;
    public:
        double getVolume(void){
            return length * breadth * height;
        }
        void setLength(double len){
            length = len;
        }
        void setBreadth(double bre){
            breadth = bre;
        }
        void setHeight(double hei){
            height = hei;
        }
        // 重载 + 运算符，用于把两个 Box 对象相加
        Box operator+(const Box& b){
            Box box;
            box.length = this->length + b.length;
            box.breadth = this->breadth + b.breadth;
            box.height = this->height + b.height;
            return box;
        }
};
// 程序的主函数
int main(){
        Box Box1;
        Box Box2;
        Box Box3;
        double volume = 0.0;

        // Box1 详述
        Box1.setLength(6.0);
        Box1.setBreadth(7.0);
        Box1.setHeight(10.0);

        // Box2 详述
        Box2.setLength(12.0);
        Box2.setBreadth(13.0);
        Box2.setHeight(10.0);

        // Box1 的体积
        volume = Box1.getVolume();
        cout << "Volume of Box1 : " << volume << endl;

        // Box2 的体积
        volume = Box2.getVolume();
        cout << "Volume of Box2 : " << volume << endl;

        // 把两个对象相加，得到 Box3
        Box3 = Box1 + Box2;
        // Box3 的体积
        volume = Box3.getVolume();
        cout << "Volume of Box3 : " << volume << endl;

        return 0;
}
*/








