/*
// 1. 宏的定义与撤销
// 普通宏定义
#define PI 3.14    // 编译阶段替换掉宏
#define T1 3+4     // 容易产生歧义
#define T2 (3+4)    // 添加括号后，语义清楚

float r = 1.0;
float area = PI * r * r;
int a = 2 * T1;    // 宏替换后变成 int a = 2*3+4  不符合本意
int a = 2 * T2;    // 宏替换后变成 int a =2*（3+4） 复合本意

#undef PI
float area = PI * r * r;    // error: 'PI' was not declared in this scope

// 引号中的宏定义不会被替换
printf("%s:%f\n", "PI", PI);     // 输出 PI: 3.14

// 宏定义的名字必须是合法标识符
#define 0x abcd      // error 不能以数字开始

// 宏定义中双引号和单引号必须成对出现
#define TEST11 "Z     // error
#define TEST2 'Z     // error


// 注意：
// （1）宏定义应注意添加括号，这样语义会比较清晰。
// （2）使用 #undef 可以撤销宏定义。
// （3）引号中的宏定义不会被替换。
// （4）宏定义的宏名必须是合法的标识符。
// （5）宏定义中单、双引号必须成对出现。

// 二、带有参数的宏定义
// max 和 min 的宏定义带参数
#define MAX(a, b) (a>b ? a:b)
#define MIN(a, b) (a<b ? a:b)

// 使用带参数的宏定义
int sum = MAX(1, 2) + MIN(1, 2);   // 替换后语句为：int sum=(1>2 ?1:2) + (1<2 ? 1:2)

// 参数个数必须宏定义时形参的个数相同
MAX(1, 2, 3);     // 会报错

#undef MAX    //  撤销MAX的宏定义
MAX(1, 2);    // error: 'MAX' was not declared in this scope

// 注意：
// （1）宏调用时参数的个数要与定义时相同。


// 三、跨行的宏定义，使用反斜杠分隔
// 定义一个交换数值的多行宏，使用反斜杠连接不同行
#define SWAP(a, b) do { \
    int t = 0;          \
    t = a;              \
    a = b;              \
    b = t;              \
} while(0)


#include <stdio.h>
#define test(x, y)       \
    do                   \
    {                    \
        int a = x;       \
        int b = y;       \
        printf("%d\n", a); \
    } while (0);         \
    int a = x + y;
int main(){
    test(1, 2);
    printf("%d\n", a);
}


// 四、三个特殊符号：****#，##，#@

#define CONNECT(a, b) a##b
#define TOCHAR(a) #@a
#define TOSTRING(a) #a 

// a##b 表示连接
int n = CONNECT(123, 456);      // 结果 n=123456
char *str = CONNECT("abcd", "efg");    // 结果 str = "abcdefg"

// @#a 表示用单引号包括参数a，返回的是一个字符
char *ch1 = TOCHAR(1);    // 结果 ch = '1'
char *ch2 = TOCHAR(123);   // 报错，单引号只用在单个字符里

// #a 表示用双引号包括参数a，返回一个字符串
char *str1 = TOSTRING(123);    // str = "123"


#include <stdio.h>
#define CONNECT(a, b) a##b
// #define TOCHAR(a) #@a
#define TOSTRING(a) #a 
int main(){
// a##b 表示连接
int n = CONNECT(123, 456);      // 结果 n=123456
// char *a = "abcd";
// char *b = "efg";
// char *str = CONNECT(a, b); // 结果 str = "abcdefg"
char *str = CONNECT("abcd", "efg"); // 结果 str = "abcdefg"

// // @#a 表示用单引号包括参数a，返回的是一个字符
// char *ch1 = TOCHAR(1);    // 结果 ch = '1'
// char *ch2 = TOCHAR(123);   // 报错，单引号只用在单个字符里

// #a 表示用双引号包括参数a，返回一个字符串
char *str1 = TOSTRING(123);    // str = "123"

    printf("n = %d\n", n);
    printf("str = %s\n", str);
    // printf("ch1 = %s\n", ch1);
    // printf("ch2 = %s\n", ch2);
    printf("str2 = %s\n", str1);
}


// 五. 常见的宏定义
// 1. 防止头文件被重复包含
#ifndef BODYDEF_H
#define BODYDEF_H
// 头文件内容
#endif
// 2. 得到指定地址上的一个字节值或字值
#include <stdio.h>
#include <stdint.h>
#define byte uint8_t
#define word int32_t
// B表示字节byte
#define MEM_B(x) (*((byte*)(x)))
// B表示字word，可以理解为int
#define MEM_W(x) (*((word*)(x)))
int main(){
    int bTest = 0x123456;
    byte m = MEM_B((&bTest));   // m=0X56
    int n = MEM_W((&bTest));     //  n=0x3456
    printf("============== %d \n", m);
    printf("============== %d \n", n);
    return 0;
}

// 3. 得到一个 field 在结构体(struct)中的偏移量
#define OFFSETOF(type, field)  ((size_t) &((type *)0)->field)
// 4. *得到一个结构体中 field 所占用的字节数
#define FSIZ(type, field) sizeof(((type *)0)->field)
// 5. 得到一个变量的地址(word宽度)
#define B_PTR(var) ((byte *)(void *) &(var))
#define W_PTR(var) ((word *)(void *) &(var))
// 6. 将一个字母转换成大写
#define UPCASE(c) (((c) >= 'a' && (c) <= 'z') ? ((c) - 0x20) : (c))
// 7. 判断字符是不是十进制的数字
#define DECCHK(c) ((c) >= '0' && (C) <= '9')
// 8. 判断字符是不是16进制的数字
#define HEXCHK(c) (((c) >= '0' && (c) <= '9') || ((c) >= 'A' && (c) <= 'F') || ((c) >= 'a' && (c) <= 'f'))
// 9. 防止溢出的一个方法
#define INC_SAT(val) (val = ((val)+1 > (val)) ? (val)+1 : (val))
// 10. 返回数组元素的个数
#define ARR_SIZE(a) (sizeof((a)) / sizeof((a[0])))


// 关于嵌套宏的使用
// 涉及到宏定义展开顺序的知识，如果宏替换以 # ##为前缀，则由外向内展开
#include <stdio.h>
#define f(x) #x   // 结果将被扩展为由实际参数替换该参数的带引号的字符串
#define b(x) a##x    // 连接实际参数
#define ac hello
int main(void){
    char *str = f(b(c));    //  display  "b(c)"
    printf("====== str = %s \n", str);
    return 0;
}

// 如果最外层 p(x) 宏替换不以# ## 为前缀，则由内向外展开
#include <stdio.h>
#define f(x) #x 
#define p(x) f(x)
#define b(x) a##x 
#define ac hello 
int main(void){
    char *str1 = p(b(c)); // display "hello"
    printf("========= str1 = %s \n", str1);
    return 0;
}


// 问题：通过宏定义实现一个可以指定前缀的字符串
// 方法1：使用#运算符。出现在宏定义中的#运算符把跟在其后的参数转换成一个字符串。有时把这种用法的 #称为字符串化运算符。
#include <stdio.h>
#define PREX 1.3.6
#define FORMAT(n) #n".%d\n"
int main(){
    int ival = 246;
    printf(FORMAT(PREX), ival);  // PREX.246
    return 0;
}
// 输出结果是：PREX.246, 和预期的结果不一样，宏PREX作为宏FORMAT的参数并没有替换。那么如何让 FORMAT 宏的参数可以替换呢？
// 首先，C语言的宏是允许嵌套的，其嵌套后，一般的展开规律像函数的参数一样；先展开参数，再分析函数，即由内向外展开。但是，注意：
// （1). 当宏中有#运算符时，参数不再被展开；
// （2). 当宏中有##运算符时，则先展开函数，再展开里面的参数；
// PS：##运算符用于把参数连接到一起。预处理程序把出现在 ## 两侧的参数合并成一个符号(非字符串)。
// 方法2：修改宏定义的格式，再添加一个中间宏 TMP(x) 实现对参数的替换，然后再替换为最终想要的字符串。

#define PREX 1.3.6
#define FORMAT(x) TMP(x)
#define TMP(x) #x".%d\n"
int main(){
    int ival = 246;
    printf(FORMAT(PREX), ival);   // 1.3.6.246
    return 0;
}


// 嵌套宏在某些情况下还是有一定的用处，但是我可能更愿意定义一个函数宏来完成上面这个工作：
#include <stdio.h>
#define FORMAT_FUN(szPrex, szFormat)                                      \
    do                                                                    \
    {                                                                     \
        char szTmp[128] = {0};                                            \
        snprintf(szTmp, sizeof(szTmp) - 1, "%s", szFormat);              \
        snprintf(szFormat, sizeof(szFormat) - 1, "%s%s", szPrex, szTmp); \
    } while (0);                                                          \

const char *szPrex = "1.3.6";
int main(){
    int ival = 246;
    char szFormat[128] = ".%d\n";
    FORMAT_FUN(szPrex, szFormat);
    // printf("%s\n", szFormat);
    printf(szFormat, ival);   // 1.3.6.246
    return 0;
}


// 举几个关于宏嵌套用法的例子
#include <stdio.h>
#define TO_STRING2(x) #x 
#define TO_STRING(x) TO_STRING1(x) 
#define TO_STRING1(x) #x 
#define PARAM(x) #x 
#define ADDPARAM(x) INT_##x 

int main(){
    const char *str = TO_STRING(PARAM(ADDPARAM(1)));
    printf("%s\n", str);
    str = TO_STRING2(PARAM(ADDPARAM(1)));
    printf("%s\n", str);
    return 0;
}


#include <stdio.h>
#define TO_STRING2(x) a_##x 
#define TO_STRING(x) TO_STRING1(x)
#define TO_STRING1(x) #x 
#define PARAM(x) #x
#define ADDPARAM(x) INT_##x 
int main(){
    const char *str = TO_STRING(TO_STRING2(PARAM(ADDPARAM(1))));
    printf("%s\n", str);
    return 0;
}
// 注意：上述例子的代码在不同的编译器上输出结果不相同。为什么呢？
// 除非替换序列中的形式参数的前面有一个#符号，或者其前面或后面有一个##符号，否则，在插入前要对宏调用的实际参数记号进行检查，并在必要时进行扩展。
// 改为：

#include <stdio.h>
#define TO_STRING2(x) a_##x 
#define TO_STRING(x) TO_STRING1(x)
#define TO_STRING1(x) T(x)
#define T(x) #x 
#define PARAM(x) #x 
#define ADDPARAM(x) INT_##x 
int main(){
    const char *str = TO_STRING(TO_STRING2(PARAM(ADDPARAM(1))));
    printf("%s\n", str);
    return 0;
}
// (1) 对TO_STRING的参数TO_STRING2(...)进行检查替换，生成标记a_PARAM(ADDPARAM(1))
// (2) 对TO_STRING1的参数a_PARAM(ADDPARAM(1))进行检查替换，生成标记a_PARAM(INT_1)
// (3) 对T的参数 a_PARAM(INT_1) 进行检查替换，生成字符串"a_PARAM(INT_1)"


#include <stdio.h>
int ival = 0;
#define A(x) printf("%d\n", ival += 1);
#define B(x) printf("%d\n", ival += 2);
#define C() printf("%d\n", ival += 3);
int main(){
    A(B(C()));
    printf("%d\n", ival);  // ?, 1
    return 0;
}

// 补充知识：
// （1）预处理器执行宏替换、条件编译以及包含指定的文件。
// （2）以#开头的命令行（"#"前可以有空格），就是预处理器处理的对象。
// （3）这些命令行的语法独立于语言的其他部分，它们可以出现在任何地方，其作用可延续到所在翻译单元的末尾（与作用域无关）。
// （4）行边界是有实际意义的。每一行都降单独进行分析。

// 关于#和##在C语言的宏中，#的功能是将其后面的宏参数进行字符串化操作（Stringfication），简单说就是在对它所引用的宏变量
// 通过替换后在其左右各加上一个双引号。比如下面代码中的宏：
#define WARN_IF(EXP)                                \
    do                                              \
    {                                               \
        if (EXP)                                    \
            fprintf(stderr, "Warning: " #EXP "\n"); \
    }                                               \
while(0)
// 那么实际使用中会出现下面所示的替换过程：
WARN_IF(divider == 0);
// 被替换为
do {
    if(divider == 0)
        fprintf(stderr, "Warning"
                        "divider == 0"
                        "/n");
} while (0);

// 这样每次 divider(除数) 为0的时候便会在标准错误流上输出一个提示信息。
// 而##被称为连接符（concatenator），用来将两个Token连接为一个 Token。注意这里连接的对象是 Token 就行，而不一定是宏的变量。
// 比如你要做一个菜单项命令名和函数指针组成的结构体的数组，并且希望在函数名和菜单项命令名之间有直观的、名字上的关系。
// 那么下面的代码就非常实用：
struct command{
    char *name;
    void (*function)(void);
};
#define COMMAND(NAME) {NAME, NAME ## _command }
// 然后你就用一些预先定义好的命令来方便的初始化一个 command 结构的数组了；
struct command commands[] = {
    COMMAND(quit),
    COMMAND(help),
    ...}
// COMMAND宏在这里充当一个代码生成器的作用，这样可以在一定程度上减少代码密度，间接地也可以减少不留心所造成的错误。
// 我们还可以n个##符号连接 n+1 个Token，这个特性也是#符号所不具备的。比如：
#define LINK_MULTIPLE(a,b,c,d) a##_##b##_##c##_##d
typedef struct _record_type LINK_MULTIPLE(name, company, position, salary);
// 这里这个语句将展开为：
// typedef struct _record_type name_company_position_salary;

// 关于 ... 的使用
// ... 在C宏中称为 Variadic Macro, 也就是变参宏，比如：
#define myprintf(templt, ...) fprintf(stderr, templt, __VA_ARGS__)
// 或者
#define myprintf(templt, args...) fprintf(stderr, templt, args)
// 第一个宏中由于没有对变参起名，我们用默认的宏 __VA_ARGS__ 来替代它。第二个宏中，我们显式地命名变参为 args, 那么我们在宏定义中就可以用
// args 来代指变参了。同 C 语言的 stdcall 一样，变参必须作为参数表的最后一项出现。当上面的宏中我们只能提供第一个参数 templt 时，C标准要求我们必须写成：
myprintf(templt, );
// 的形式。这时的替换过程为：
myprintf("Error!\n", );
// 替换为：
fprintf(stderr, "Error!\n",);
// 这是一个语法错误，不能正常编译。这个问题一般有两个解决方法。首先，GUN CPP提供的解决方法允许上面的宏调用写成：
myprintf(templt);
// 而它将会被通过替换变成：
fprintf(stderr, "Error!\n", );
// 很明显，这里仍然会产生编译错误（非本例的某些情况下不会产生编译错误）。除了这种方式外，c99和GUN CPP 都支持下面的宏定义方式：
#define myprintf(templt, ...) fprintf(stderr, templt, ##__VAR_ARGS__)
// 这时，##这个连接符号充当的作用就是当 __VAR_ARGS__ 为空的时候，消除前面的那个逗号。那么此时的翻译过程如下：
myprintf(templt);
// 被转化为：
fprintf(stderr, templt);
// 这样如果 templt 合法，将不会产生编译错误。


// 错误的嵌套 - Misnesting
// 宏的定义不一定要有完整的、配对的括号，但是为了避免出错并且提高可读性，最好避免这样使用。
// 由操作符优先级引起的问题 - Operator Precedence Problem
// 由于宏只是简单的替换，宏的参数如果是复合结构，那么通过替换之后可能由于各个参数之间的操作符优先级高于单个参数内部各部分之间
// 相互作用的操作符优先级，如果我们不用括号保护各个宏参数，可能会产生预想不到的情形。比如：
#define ceil_div(x, y) (x + y - 1) / y
// 那么
a = ceil_div(b & c, sizeof(int));
// 将被转化为：
a = (b & c + sizeof(int) - 1) / sizeof(int);
// 由于 +/- 的优先级高于 & 的优先级，那么上面式子等同于：
a = (b & (c + sizeof(int) - 1)) / sizeof(int);
// 为了避免，需要多加几个括号
#define ceil_div(x, y) (((x) +(y)-1)/(y))


// 消除多余的分号 - Semicolon Swallowing
// 通常情况下，为了使函数摸样的宏在表面上看起来像一个通常的C语言调用一样，通常情况下我们在宏的后面加上一个分号，比如下面的带参宏：
MY_MACRO(x);
// 但是如果是下面的情况：
#define MY_MACRO(X) { \
/* line 1  */     \
/* line 2   */    \ }

// ...

if (condition())
    MY_MACRO(a);
else
    {...}
// 这样会由于多出的那个分号产生编译错误。为了避免这种情况出现同时保持 MY_MACRO(x)；的这种写法，我们需要把宏定义为这种形式：
#define MY_MACRO(x) do {
    /* line 1 */ \
    /* line 2 */ \
    /* line 3 */ \
} while(0)
// 这样只要保证总是使用分号，就不会有任何问题。
* /

// Duplication of Side Effects
// 这里的 Side Effect 是指宏在展开的时候对其参数可能进行多次 Evaluation(也就是取值)，但是如果这个宏参数是一个函数，那么就有可能
// 被调用多次从而达到不一致的结果，甚至会发生更严重的错误。比如：
#define min(X, Y) ((X) > (Y) ? (Y) : (X))
    // ...
c = min(a, foo(b));
// 这时 foo() 函数就被调用了两次。为了解决这个潜在的问题，我们应当这样写 min(X, Y)这个宏：
#define min(X, Y) ({ \
typeof (X) x_ = (X); \
typeof (Y) y_ = (Y); \
(x_ < y_) ? x_ : y_;})
// ({...}) 的作用是将内部的几条语句中的最后一条的值返回，它也允许在内部声明变量（因为它通过大括号组成了一个局部Scope）
