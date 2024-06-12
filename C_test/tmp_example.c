/*   按位与 &
#include <stdio.h>
int main(){
    int a, b;
    a = 077;
    printf("==== a = %d \n", a);
    b = a & 3;
    printf("a & b(decimal) 为 %d \n", b);
    b &= 7;
    printf("a & b(decimal) 为 %d \n", b);
    return 0;
}

// 按位或 |
#include <stdio.h>
int main(){
    int a, b;
    a = 077;
    b = a | 3;
    printf("b 的值为 %d \n", b);
    b |= 7;
    printf("b 的值为 %d \n", b);
    return 0;
}


// 按位异或 ^, 0^0=0; 0^1=1; 1^0=1; 1^1=0 
#include <stdio.h>
int main(){
    int a, b;
    a = 077;
    b = a ^ 3;
    printf("b 的值为 %d \n", b);
    b ^= 7;
    printf("b 的值为 %d \n", b);
    return 0;
}


#include <stdio.h>
int main(){
    unsigned a, b, c, d;
    printf("请输入整数：\n");
    scanf("%o", &a);
    b = a >> 4;
    printf("输出整数 b = %d \n", b);
    int tmp_a, tmp_b, tmp_c;
    tmp_a = ~0;
    tmp_b = tmp_a << 4;
    tmp_c = ~tmp_b;
    printf("输出整数 tmp_a = %d \n", tmp_a);
    printf("输出整数 tmp_b = %d \n", tmp_b);
    printf("输出整数 tmp_c = %d \n", tmp_c);
    c = ~(~0 << 4); // 设置一个低 4 位全为 1，其余全为 0 的数，可用~(~0<<4)
    printf("输出整数 c = %d \n", c);
    d = b & c;
    printf("%o\n%o\n", a, d);
    return 0;
}


// 按位取反 ~ 55 补码，反码，原码
#include <stdio.h>
int main(){
    int a, b;
    a = 234;
    b = ~a;
    printf("a 的按位取反值为（十进制）%d \n", b);
    a = ~a;
    printf("a 的按位取反值为（十六进制）%x \n", a);
    return 0;
}


#include <graphics.h>
int main(){
    int driver, mode, i;
    float j = 1, k = 1;
    driver = VGA;
    mode = VGAHI;
    initgraph(&driver, &mode, "");
    setbkcolor(YELLOW);
    for (i = 0; i <= 25; i++){
        setcolor(8);
        circle(310, 250, k);
        k = k + j;
        j = j + 0.3;
    }
    return 0;
}


// 打印杨辉三角 61
#include <stdio.h>
int main(){
    int i, j;
    int a[10][10];
    printf("\n");
    for (i = 0; i < 10; i++){
        a[i][0] = 1;
        a[i][i] = 1;
    }
    for (i = 2; i < 10; i++)
        for (j = 1; j < i; j++)
            a[i][j] = a[i - 1][j - 1] + a[i - 1][j];
    for (i = 0; i < 10;i++){
        for (j = 0; j <= i; j++)
            printf("%5d", a[i][j]);
        printf("\n");
    }
    return 0;
}


// 66，三个数大小排序，利用指针方法
#include <stdio.h>
void swap(int *, int *);
int main(void){
    int a, b, c;
    int *p1, *p2, *p3;
    printf("输入 a, b, c :\n");
    scanf("%d %d %d", &a, &b, &c);
    p1 = &a;
    p2 = &b;
    p3 = &c;
    if (a>b)
        swap(p1, p2);
    if (a>c)
        swap(p1, p3);
    if (b>c)
        swap(p2, p3);
    printf("%d %d %d\n", a, b, c);
}
void swap(int *s1, int *s2){
    int t;
    t = *s1;
    *s1 = *s2;
    *s2 = t;
}


// 67 输入数组，最大值与第一个元素交换，最小值与最后一个交换，输出数组
#include <stdio.h>
#include <stdlib.h>
void fun(int *s, int n){
    int i;
    int max = s[0];
    int a = 0;
    for (i = 0; i < n; i++){
        if(s[i]>max){
            max = s[i];
            a = i;
        }
    }
    s[a] = s[0];
    s[0] = max;
    int j;
    int min = s[n - 1];
    int b = n - 1;
    for (j = 0; j < n;j++){
        if(s[j]<min)
        {
            min = s[j];
            b = j;
        }
    }
    s[b] = s[n - 1];
    s[n - 1] = min;
}
void printf_s(int *s, int n){
    int i;
    for (i = 0; i < n;i++)
        printf("%d ", s[i]);
    printf("\n");
}
int main(){
    int s[20];
    int i, n;
    printf("设置数组长度(<20):");
    scanf("%d", &n);
    printf("输入 %d 个元素：\n", n);
    for (i = 0; i < n;i++)
        scanf("%d", &s[i]);
    fun(s, n);
    printf_s(s, n);
    return 0;
}


// 68 有n个整数，前面各数后移m位置，最后m个变成前面m个
#include <stdio.h>
void shiftArray(int arr[], int n, int m){
    int temp[m];
    // 保存最后 m个数到临时数组
    for (int i = n - m, j = 0; i < n; i++, j++){
        temp[j] = arr[i];
    }
    // 将前面的 n-m 个数向后移动m个位置
    for (int i = n - m - 1; i >= 0; i--){
        arr[i + m] = arr[i];
    }
    // 将临时数组中的数放到最前面
    for (int i = 0; i < m; i++){
        arr[i] = temp[i];
    }
}
int main(){
    int n, m;
    printf("请输入整数个数 n: ");
    scanf("%d", &n);
    printf("请输入向后移动的位置 m: ");
    scanf("%d", &m);
    int arr[n];
    printf("请输入 %d 个整数：", n);
    for (int i = 0; i < n; i++){
        scanf("%d", &arr[i]);
    }
    shiftArray(arr, n, m);
    printf("移动后的数组：");
    for (int i = 0; i < n; i++){
        printf("%d ", arr[i]);
    }
    printf("\n");
    return 0;
}

// 使用一个辅助数组来保存需要移动的元素，然后再将元素按照要求移动到正确的位置
#include <stdio.h>
#include <stdlib.h>
int main(){
    int arr[6];
    int i, n, offset;
    // 输入数组大小和数组内容
    printf("Total numbers? \n");
    scanf("%d", &n);
    printf("Input %d numbers.\n", n);
    for (i = 0; i < n; i++)
        scanf("%d", &arr[i]);
    // 输入滚动偏移量
    printf("Set your offset.\n");
    scanf("%d", &offset);
    printf("Offset is %d. \n", offset);
    // 打印滚动前数组
    print_arr(arr, n);
    // 滚动数组并打印
    move(arr, n, offset);
    print_arr(arr, n);
    return 0;
}
void print_arr(int array[], int n){
    int i;
    for (i = 0; i < n;++i)
        printf("%4d", array[i]);
    printf("\n");
}
// 滚动数组
void move(int array[], int n, int offset){
    int *p, *arr_end;
    arr_end = array + n;   // 数组最后一个元素的下一个位置
    int last;
    // 滚动直到偏移量为0
    while(offset){
        last = *(arr_end - 1);
        for (p = arr_end - 1; p != array; --p)  // 向右滚动一位
            *p = *(p - 1);
        *array = last;
        --offset;
    }
}


#include <stdio.h>
int main(){
    int arr[5] = {1, 2, 3, 4, 5};
    int *p, *arr_end;
    for (int i = 0; i < 5; i++){
        printf(" %d ", arr[i]);
    }
    printf("\n");
    arr_end = arr + 3;
    for (int i = 0; i < 5; i++){
        // printf("  %d  ", *(arr_end-i));
        printf("  %d  ", arr_end[i]);
    }
    printf("\n");
    for (int i = 0; i < 5; i++)
    {
        printf(" %d ", *(arr-i));
    }
    return 0;
}


// 69，有n个人围成一圈，顺序排号。从第一个人开始报数（1到3报数），凡报到3的人退出圈子，问最后留下的是原来第几号的那位。
#include <stdio.h>
int main(){
    int num[50], n, *p, j, loop, i, m, k;
    printf("请输入这一圈人的数量：\n");
    scanf("%d", &n);
    p = num;
    // 开始给这些人编号
    for (j = 0; j < n; j++){
        *(p + j) = j + 1;
    }
    i = 0;  // i用于计数，即让指针后移
    m = 0;  // m记录退出圈子的人数
    k = 0;  // k报数1，2，3
    while(m<n-1){ //不能写成m<n,因为假设有8人，当退出了6人时，此时还是进行人数退出，即m++，如果是7<8,剩下一个人喊1，2，3，然后退出，将不会有输出
        if(*(p+i)!=0) // 如果这个人的头上编号不是0就开始报数
        {
            k++;
        }
        if(k==3){
            k = 0;  // 报数清零，即下一个人从1开始报数
            *(p + i) = 0;  // 将报数为3的人编号重置为0
            m++;
        }
        i++;
        if(i==n){
            i = 0;
        }
    }
    printf("现在剩下的人是：");
    for (loop = 0; loop < n; loop++){
        if(num[loop] != 0){
            printf("%2d 号\n", num[loop]);
        }
    }
}

// 求字符串的长度
#include <stdio.h>
#include <stdlib.h>
int length(char *s);
int main(){
    char str[100];  // 可以根据实际情况增大数组长度
    printf("请输入字符串：\n");
    scanf("%s", str);
    int len = length(str);   // 调用 length 函数计算字符串长度
    printf("字符串有 %d 个字符。\n", len);
    return EXIT_SUCCESS;
}
// 求字符串长度
int length(char *s){
    int i = 0;
    while(*s != '\0'){
        i++;
        s++;
    }
    return i;
}


// 71. 编写 input() 和 output() 函数输入，输出5个学生的数据记录
#include <stdio.h>
#include <stdlib.h>
typedef struct{
    char name[20];
    char sex[5];
    int age;
} Stu;
void input(Stu *stu);
void output(Stu *stu);
int main(){
    Stu stu[5];
    printf("请输入5个学生的信息: 姓名 性别 年龄：\n");
    input(stu);
    printf("5个学生的信息如下: \n姓名 性别 年龄\n");
    output(stu);
    return 0;
}
void input(Stu *stu){
    int i;
    for (i = 0; i < 5; i++)
        scanf("%s%s%d", stu[i].name, stu[i].sex, &(stu[i].age));
}
void output(Stu *stu){
    int i;
    for (i = 0; i < 5; i++)
        printf("%s %s %d\n", stu[i].name, stu[i].sex, stu[i].age);
}


// 72. 创建一个链表
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
typedef struct LNode{
    int data;
    struct LNode *next;
} LNode, *LinkList;
LinkList CreateList(int n);
void print(LinkList h);
int main(){
    LinkList Head = NULL;
    int n;
    scanf("%d", &n);
    Head = CreateList(n);
    printf("刚刚建立的各个链表元素的值为：\n");
    print(Head);
    printf("\n\n");
    return 0;
}
LinkList CreateList(int n){
    LinkList L, p, q;
    int i;
    L = (LNode *)malloc(sizeof(LNode));
    if(!L)
        return 0;
    L->next = NULL;
    q = L;
    for (i = 1; i <= n;i++){
        p = (LinkList)malloc(sizeof(LNode));
        printf("请输入第%d个元素的值:", i);
        scanf("%d", &(p->data));
        p->next = NULL;
        q->next = p;
        q = p;
    }
    return L;
}
void print(LinkList h){
    LinkList p = h->next;
    while(p!=NULL){
        printf("%d ", p->data);
        p = p->next;
    }
}


// 73. 反向输出一个链表
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

typedef struct LNode{
    int data;
    struct LNode *next;
} LNode, *LinkList;
LinkList CreateList(int n);
void print(LinkList h);
int main(){
    LinkList Head = NULL;
    int n;
    scanf("%d", &n);
    Head = CreateList(n);
    printf("刚刚建立的各个链表元素的值为：\n");
    print(Head);
    printf("\n\n");
    return 0;
}
LinkList CreateList(int n){
    LinkList L, p, q;
    int i;
    L = (LNode *)malloc(sizeof(LNode));
    if(!L)
        return 0;
    L->next = NULL;
    q = L;
    for (i = 1; i <= n; i++){
        p = (LinkList)malloc(sizeof(LNode));
        printf("请输入第%d个元素的值: ", i);
        scanf("%d", &(p->data));
        p->next = NULL;
        q->next = p;
        q = p;
    }
    return L;
}
void print(LinkList h){
    LinkList p = h->next;
    while(p!=NULL){
        printf("%d", p->data);
        p = p->next;
    }
}


// 74. 连接两个链表
#include <stdlib.h>
#include <stdio.h>
struct list{
    int data;
    struct list *next;
};
typedef struct list node;
typedef node *link;
link delete_node(link pointer, link tmp)
{
    if (tmp==NULL)  // delete first node
        return pointer->next;
    else{
        if(tmp->next->next==NULL)  // delete last node
            tmp->next = NULL;
        else  // delete the other node
            tmp->next = tmp->next->next;
        return pointer;
    }
}
void selection_sort(link pointer, int num){
    link tmp, btmp;
    int i, min;
    for (i = 0; i < num;i++){
        tmp = pointer;
        min = tmp->data;
        btmp = NULL;
        while(tmp->next){
            if (min>tmp->next->data)
            {
                min = tmp->next->data;
                btmp = tmp;
            }
            tmp = tmp->next;
        }
        printf("\40: %d\n", min);
        pointer = delete_node(pointer, btmp);
    }
}
link create_list(int array[], int num){
    link tmp1, tmp2, pointer;
    int i;
    pointer = (link)malloc(sizeof(node));
    pointer->data = array[0];
    tmp1 = pointer;
    for (i = 1; i < num;i++){
        tmp2 = (link)malloc(sizeof(node));
        tmp2->next = NULL;
        tmp2->data = array[i];
        tmp1->next = tmp2;
        tmp1 = tmp1->next;
    }
    return pointer;
}
link concatenate(link pointer1, link pointer2){
    link tmp;
    tmp = pointer1;
    while(tmp->next)
        tmp = tmp->next;
    tmp->next = pointer2;
    return pointer1;
}
int main(void){
    int arr1[] = {3, 12, 8, 9, 11};
    link ptr;
    ptr = create_list(arr1, 5);
    selection_sort(ptr, 5);
    return 0;
}


// 75. 输入一个整数，并将其反转后输出
#include <stdio.h>
int main(){
    int n, reversedNumber = 0, remainder;
    printf("请输入一个整数: ");
    scanf("%d", &n);
    while(n!=0){
        remainder = n % 10;
        reversedNumber = reversedNumber * 10 + remainder;
        n /= 10;
    }
    printf("反转后的整数：%d", reversedNumber);
    return 0;
}


// 76. 编写一个函数，输入n为偶数时，调用1/2+1/4+..+1/n, 输入n为奇数时，调用函数1/1+1/3+..+1/n(利用指针函数)
#include <stdio.h>
#include <stdlib.h>
double evenumber(int n);
double oddnumber(int n);
int main(){
    int n;
    double r;
    double (*pfunc)(int);
    printf("请输入一个数字：");
    scanf("%d", &n);
    if(n%2==0)
        pfunc = evenumber;
    else
        pfunc = oddnumber;
    r = (*pfunc)(n);
    printf("%lf\n", r);
    return 0;
}
double evenumber(int n){
    double s = 0, a = 0;
    int i;
    for (i = 2; i <= n; i+=2){
        a = (double)1 / i;
        s += a;
    }
    return s;
}
double oddnumber(int n){
    double s = 0, a = 0;
    int i;
    for (i = 1; i <= n; i+=2){
        a = (double)1 / i;
        s += a;
    }
    return s;
}


// 77. 填空练习（指向指针的指针）
#include <stdio.h>
#include <stdlib.h>
int main(){
    const char *s[] = {"man", "woman", "girl", "boy", "sister"};
    const char **q;
    int k;
    for (k = 0; k < 5; k++){
        q = &s[k];    // 这里填入内容
        printf("%s\n", *q);
    }
    return 0;
}


// 78. 找到年龄最大的人，并输出。
#include <stdio.h>
#include <stdlib.h>
struct man{
    char name[20];
    int age;
} 
person[3] = {"li", 18, "wang", 25, "sun", 22};
int main(){
    struct man *q, *p;
    int i, m = 0;
    p = person;
    for (i = 0; i < 3;i++){
        if(m<p->age){
                m = p->age;
                q = p;
        }
         p++;
    }
    printf("%s %d\n", q->name, q->age);
    return 0;
}


// 79.字符串排序
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void swap(char *str1, char *str2);
int main(){
    char str1[20], str2[20], str3[20];
    printf("请输入三个字符串，每个字符串以回车结束！:\n");
    fgets(str1, (sizeof str1 / sizeof str1[0]), stdin);
    fgets(str2, (sizeof str2 / sizeof str2[0]), stdin);
    fgets(str3, (sizeof str3 / sizeof str3[0]), stdin);
    if(strcmp(str1, str2)>0)
        swap(str1, str2);
    if (strcmp(str2, str3)>0)
        swap(str2, str3);
    if(strcmp(str1,str2)>0)
        swap(str1, str2);
    printf("排序后的结果为：\n");
    printf("%s\n%s\n%s\n", str1, str2, str3);
    return 0;
}
void swap(char *str1, char* str2){
    char tem[20];
    strcpy(tem, str1);
    strcpy(str1, str2);
    strcpy(str2, tem);
}


// 80. 海滩上一堆桃子，五个猴子分，第一个分成五份，多一个，扔海里，第二个按上述分剩下的，后续一样
#include <stdio.h>
#include <stdlib.h>
int main(){
    int x, i = 0, j = 1;
    while(i<5){
        x = 4 * j;
        for (i = 0; i < 5;i++){
            if(x%4 != 0){
                break;
            }
            x = (x / 4) * 5 + 1;
        }
        j++;
    }
    printf("%d\n", x);
    return 0;
}


#include <stdio.h>
#include <stdlib.h>
int calc_peaches(int monkeys){
    // 猴子总数
    int m = monkeys;
    // 猴子每次待分桃子数
    int remain;
    int i = 1;
    while(1){
        // 最后一只猴子待分配的桃子数一定能被4整除
        remain = 4 * i;
        for (m = monkeys; m > 0; m--){
            // 每只猴子待分配的桃子数remain，满足（remain-1）% 5 = 0
            if((remain - 1) % 5 != 0)
                break;
            // 除第一只猴子外，其他猴子待分配的桃子数 remain，满足 remain % 4 = 0
            if (m>1 && remain % 4 != 0)
                break;
            // 不是第一只猴子时，计算出上一只猴子待分配的桃子数
            if (m > 1)
                remain = remain / 4 * 5 + 1;
            else
                break;
        }
        printf("============= the %d run after for loop ====\n", m);
        // 待分配的桃子已满足 monkeys 只猴子的分配要求
        if(m == 1)
            break;
        // 不满足时，则说明最后一只猴子待分配的桃子数不对，更换后再试
        else
            i++;
    }
    return remain;
}
int main(void){
    int monkeys;
    printf("请输入猴子的数量：");
    scanf("%d", &monkeys);
    printf("桃子总数：%d\n", calc_peaches(monkeys));
}


// 设最少有 x 个桃子
// 第一只 4/5(x-1) = a

// 81. 809*??=800*??+9*??，其中??代表的两位数，809*??为四位数，8*??的结果为两位数，9*??的结果为3位数。求??代表的两位数，及809*??后的结果。
#include <stdio.h>
void output(long int b, long int i){
    printf("\n%ld = 800 * %ld + 9 * %ld\n", b, i, i);
}
int main(){
    void output(long int b, long int i);
    long int a, b, i;
    a = 809;
    for (i = 10; i < 100; i++){
        b = i * a;
        if(b>= 1000 && b <= 10000 && 8*i<100 && 9*i >= 100){
            output(b, i);
        }
    }
    return 0;
}


// 82. 八进制转换为十进制
#include <stdio.h>
#include <stdlib.h>
int main()
{
    int n = 0, i = 0;
    char s[20];
    printf("请输入一个8进制数:\n");
    gets(s);
    while(s[i]!='\0'){
        n = n * 8 + s[i] - '0';
        i++;
    }
    printf("刚输入的8进制数转化为十进制为\n%d\n", n);
    return 0;
}


#include <stdio.h>
#include <math.h>
int OtoD(int n){
    int sum = 0;
    int i = 0;
    while(n){
        sum += (n % 10) * pow(8, i++);
        n /= 10;
    }
    return sum;
}
int main(void){
    int n;
    printf("请输入一个8进制数:\n");
    scanf("%d", &n);
    printf("刚输入的8进制数转化为十进制为\n");
    printf("%d", OtoD(n));
}


// 83. 求0-7所能组成的奇数个数
#include <stdio.h>
int main(int argc, char *argv[]){
    long sum = 4, s = 4; // sum 的初始值为4表示，只有一位数字组成的奇数个数为4个
    int j;
    for (j = 2; j <= 8; j++){
        printf("%d位数为奇数的个数%ld\n", j - 1, s);
        if (j<=2)
            s *= 7;
        else
            s *= 8;
        sum += s;
    }
    printf("%d位数为奇数的个数%ld\n", j - 1, s);
    printf("奇数的总个数为：%ld\n", sum);
    return 0;
}


// 84. 一个偶数总能表示为两个素数之和
#include <stdio.h>
#include <math.h>
int IsPrime(int n){
    int i;
    if(n==1)
        return 0;
    for (i = 2; i <= sqrt(n); i++)
        if(n%i == 0)
            return 0;
    return 1;
}
void divide_even(int even, int *a, int *b){
    int i;
    for (i = 2; i < even; i++){
        if(IsPrime(i)&&IsPrime(even - i))
            break;
    }
    *a = i;
    *b = even - i;
}
int main(void){
    int n, a, b;
    printf("请输入一个偶数：\n");
    scanf("%d", &n);
    divide_even(n, &a, &b);
    printf("偶数%d可以分解成%d和%d两个素数的和", n, a, b);
}


// 85. 判断一个素数能被几个9整除
#include <stdio.h>
#include <stdlib.h>
int main(){
    int p, i;
    long int sum = 9;
    printf("请输入一个素数：\n");
    scanf("%d", &p);
    for (i = 1;; i++)
        if(sum%p==0)
            break;
        else
            sum = sum * 10 + 9;
    printf("素数%d能整除%d个9组成的整数%ld\n", p, i, sum);
    return 0;
}


// 86. 两个字符串连接程序
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
char *strconnect(char *str1, char *str2);
int main(){
    char str1[20], str2[20];
    char *str;
    puts("请输入两个字符串，用回车分开：");
    scanf("%s%s", str1, str2);
    str = strconnect(str1, str2);
    puts("连接后的字符串为：");
    puts(str);
    return 0;
}
char* strconnect(char *str1, char *str2){
    char *str;
    str = (char *)malloc(strlen(str1) + strlen(str2) + 1);
    str[0] = '\0';
    strcat(str, str1);
    strcat(str, str2);
    return str;
}


// 87. 回答结果（结构体变量传递）
#include <stdio.h>
struct student{
    int x;
    char c;
} a;
int main(){
    a.x = 3;
    a.c = 'a';
    f(a);
    printf("%d, %c", a.x, a.c);
}
f(struct student b){
    b.x = 20;
    b.c = 'y';
}   //  值传递，结果不会变

// 修改为址传递
#include <stdio.h>
struct student{
    int x;
    char c;
} a;
struct student f(struct student *b);   // 函数声明
int main(){
    a.x = 3;
    a.c = 'a';
    f(&a);
    printf("%d, %c", a.x, a.c);
    return 0;
}
struct student f(struct student *b){
    b->x = 20;
    b->c = 'y';
}


// 88. 读取7个数（1-50）的整数值，每读取一个值，程序打印出该值个数的 *
#include <stdio.h>
#include <stdlib.h>
int main(){
    int n, i, j;
    printf("请输入数字：\n");
    i--;
    for (i = 0; i < 7; i++){
        scanf("%d", &n);
        if(n>50){
            printf("请重新输入：\n");
            i--;
        } else {
            for (j = 0; j < n; j++)
                printf("*");
        }
        printf("\n");
    }
    return 0;
}


// 89. 某个公司采用公用电话传递数据，数据是四位的整数，在传递过程中是加密的，加密规则如下：
// 每位数字都加上5，然后用和除以10的余数代替该数字，再将第一位和第四位交换，第二位和第三位交换。
#include <stdio.h>
int main(){
    int a, i, aa[4], t;
    printf("请输入四位数字：");
    scanf("%d", &a);
    aa[0] = a % 10;
    aa[1] = a % 100 / 10;
    aa[2] = a % 1000 / 100;
    aa[3] = a / 1000;
    for (i = 0; i <= 3; i++){
        aa[i] += 5;
        aa[i] %= 10;
    }
    for (i = 0; i <= 3 / 2; i++){
        t = aa[i];
        aa[i] = aa[3 - i];
        aa[3 - i] = t;
    }
    printf("加密后的数字：");
    for (i = 3; i >= 0; i--)
        printf("%d", aa[i]);
    printf("\n");
}


// 90. 专升本，反转数组，读结果
#include <stdio.h>
#include <stdlib.h>
#define M 5
int main(){
    int a[M] = {1, 2, 3, 4, 5};
    int i, j, t;
    i = 0;
    j = M - 1;
    while(i<j){
        t = *(a + i);
        *(a + i) = *(a + j);
        *(a + j) = t;
        i++;
        j--;
    }
    // char ch = getchar();   // 使用该方式打断点使用
    for (i = 0; i < M; i++)
    {
        printf("%d\n", *(a + i));
    }
}


// 91. 时间函数举例1
#include <stdio.h>
#include <time.h>
int main(){
    time_t rawtime;
    struct tm *timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    printf("当前本地时间为：%s", asctime(timeinfo));
    return 0;
}


// 92. 时间函数举例2
#include <stdio.h>
#include <time.h>
#include <unistd.h>
int main(){
    time_t start, end;
    int i;
    int res;
    start = time(NULL);
    for (i = 0; i < 300000; i++){
        // printf("\n");   // 返回两个 time_t 型变量之间的时间间隔
        res++;
        usleep(1);   // 在 Linux 上实现sleep功能，你可以使用 sleep() 函数，位于 unistd.h 头文件中。
    }     //  sleep(5) 会使程序暂停5秒，usleep() 函数的参数是微秒
    end = time(NULL);
    // 输出执行时间
    printf("时间间隔为%6.3f\n", difftime(end, start));
    return 0;
}


// 93. 时间函数举例2
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main(){
    long i = 10000000L;
    clock_t start, finish;
    double TheTimes;
    printf("做%ld次空循环需要的时间为", i);
    start = clock();
    while(i--)
        ;
    finish = clock();
    TheTimes = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("%f秒。\n", TheTimes);
    return 0;
}

// 94. 猜谜游戏
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void caizi(void){
    int n;
    char begin;
    int count = 1;
    srand((int)time(NULL));
    int m = (rand() % 100) + 1;
    puts("游戏开始，请输入数字：");
    while(1){
        scanf("%d", &n);
        if(n==m){
            printf("猜中了，使用了 %d 次！\n", count);
            if (count == 1){
                printf("你是神级人物！膜拜\n");
                getchar();
                printf("你已经达到最高级别，还需要玩吗？Y/N \n");
                scanf("%c", &begin);
                if (begin == 'Y' || begin == 'y')  // 重复玩的一个嵌套循环
                {
                    caizi();
                } else {
                    printf("谢谢，再见!\n");
                }
            } else if (count <=5){
                printf("你是王级人物！非常赞\n");
                getchar();
                printf("需要挑战最高级别不？Y/N \n");
                scanf("%c", &begin);
                if(begin == 'Y' || begin == 'y'){
                    caizi();
                } else{
                    printf("谢谢，再见！\n");
                }
            } else if(count <= 10){
                printf("你是大师级人物了！狂赞\n");
                getchar();
                printf("需要挑战最高级别不？Y/N \n");
                scanf("%c", &begin);
                if(begin == 'Y' || begin == 'y'){
                    caizi();
                } else {
                    printf("谢谢，再见！\n");
                }
            } else if(count <= 15){
                printf("你是钻石级人物了！怒赞\n");
                getchar();
                printf("需要挑战更高级别不？Y/N \n");
                scanf("%c", &begin);
                if(begin == 'Y' || begin == 'y'){
                    caizi();
                } else{
                    printf("谢谢，再见！\n");
                }
            } else {
                getchar();
                printf("你的技术还有待提高哦！重玩？Y/N\n");
                scanf("%c", &begin);
                if(begin =='Y' || begin == 'y'){
                    caizi();
                } else {
                    printf("谢谢，再见！\n");
                }
            }
            break;
        }
        else if (n< m)
        {
            puts("太小了！");
            puts("重新输入：");
        }else {
            puts("太大了！");
            puts("重新输入：");
        }
        count++;  // 计数器
    }
}
int main(void){
    caizi();
    return 0;
}


// 95. 简单的结构体应用实例
#include <stdio.h>
struct programming{
    float constant;
    char *pointer;
};
int main(){
    struct programming variable;
    char string[] = "菜鸟教程：https://www.runoob.com";
    variable.constant = 1.23;
    variable.pointer = string;
    printf("%f\n", variable.constant);
    printf("%s\n", variable.pointer);
    return 0;
}


// 96. 计算字符串子串出现的次数
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(){
    int i, j, k, TLen, PLen, count = 0;
    char T[50], P[10];
    printf("请输入两个字符串，以回车隔开，母串在前，子串在后：\n");
    gets(T);
    gets(P);
    TLen = strlen(T);
    PLen = strlen(P);
    for (i = 0; i <= TLen - PLen; i++){
        for (j = 0, k = i; j < PLen && P[j] == T[k]; j++, k++)
            ;
        if(j==PLen)
            count++;
    }
    printf("%d\n", count);
    return 0;
}


// 97. 从键盘输入一些字符，逐个把它们送到磁盘上去，直到输入一个#为止
#include <stdio.h>
#include <stdlib.h>
int main(){
    FILE *fp = NULL;
    char filename[25];
    char ch;
    printf("输入你要保存到的文件的名称：\n");
    gets(filename);
    if ((fp=fopen(filename,"w"))==NULL){
        printf("error: cannot open file!\n");
        exit(0);
    }
    printf("现在你可以输入你要保存的一些符号，以#结束：\n");
    getchar();
    while((ch=getchar()) != '#'){
        fputc(ch, fp);
    }
    fclose(fp);
    return 0;
}


// 98. 从键盘输入一个字符串，将小写字母全部转换成大写字母，然后输出到一个磁盘文件"test"中保存。输入的字符串以！结束
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(){
    FILE *fp = NULL;
    char str[50];
    int i, len;
    printf("输入一个字符串：\n");
    gets(str);
    len = strlen(str);
    for (i = 0; i < len; i++){
        if(str[i]<='z'&&str[i]>='a')
            str[i] -= 32;
    }
    if((fp=fopen("test", "w"))==NULL){
        printf("error: cannot open file!\n");
        exit(0);
    }
    fprintf(fp, "%s", str);
    fclose(fp);
    return 0;
}


// 99. 有两个磁盘文件A和B，各存放一行字母，要求把这两个文件中的信息合并(按字母顺序排列)，输出到一个新文件C中。
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(){
    FILE *fa, *fb, *fc;
    int i, j, k;
    char str[100], str1[100];
    char tem;
    if((fa=fopen("A.txt", "r"))==NULL)  // A.txt 文件需要存在
    {
        printf("error: cannot open A file!\n");
        exit(0);
    }
    fgets(str, 99, fa);
    fclose(fa);
    if((fb=fopen("B.txt", "r")) == NULL)    // B.txt 文件需要存在
    {
        printf("error: cannot open B file!\n");
        exit(0);
    }
    fgets(str1, 100, fb);
    fclose(fb);
    strcat(str, str1);
    for (i = strlen(str) - 1; i > 1; i--)
        for (j = 0; j < i;j++)
            if(str[j]>str[j+1]){
                tem = str[j];
                str[j] = str[j + 1];
                str[j + 1] = tem;
            }
    if((fc=fopen("C.txt", "w"))==NULL) // 合并为 C.txt
    {
        printf("error: cannot open C file!\n");
        exit(0);
    }
    fputs(str, fc);
    fclose(fc);
    return 0;
}


// 100. 有五个学生，每个学生有三门成绩，从键盘上输入以上数据（包括学生号，姓名，三门课成绩），计算出平均成绩
// 况原有的数据和计算出的平均分数存放在磁盘文件"stud"中
#include <stdio.h>
#include <stdlib.h>
typedef struct{
    int ID;
    int math;
    int English;
    int C;
    int avargrade;
    char name[20];
} Stu;
int main(){
    FILE *fp;
    Stu stu[5];
    int i, avargrade = 0;
    printf("请输入5个同学的信息：学生号，姓名，3门成绩：\n");
    for (i = 0; i < 5;i++){
        scanf("%d %s %d %d %d", &(stu[i].ID), stu[i].name, &(stu[i].math), &(stu[i].English), &(stu[i].C));
        stu[i].avargrade = (stu[i].math + stu[i].English + stu[i].C) / 3;
    }
    if((fp=fopen("stud", "w"))==NULL){
        printf("error:cannot open file!\n");
        exit(0);
    }
    for (i = 0; i < 5;i++)
        fprintf(fp, "%d %s %d %d %d %d\n", stu[i].ID, stu[i].name, stu[i].math, stu[i].English, stu[i].C, stu[i].avargrade);

    fclose(fp);
    return 0;
}
*/


