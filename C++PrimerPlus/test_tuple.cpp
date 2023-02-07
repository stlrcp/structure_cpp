/*
//     https://blog.csdn.net/weixin_55664293/article/details/125365041     //
// tuple 对象的创建和初始化
#include <iostream>
#include <tuple>

int main() {
    std::tuple<std::string, int> t1;    // 无参构造
    std::tuple<std::string, int> t2(t1); // 拷贝构造
    std::tuple<std::string, int> t3(std::make_tuple("Lily", 1));
    std::tuple<std::string, long> t4(t3);    // 隐式类型转换构造的左值类型
    std::tuple<std::string, int> t5("Mike", 2);  // 初始化列表构造的右值模式
    std::tuple<std::string, int> t6(std::make_pair("Jack", 3));   // 将pair对象转换为tuple对象
    std::cout << std::get<0>(t5) << std::endl;
    std::cout << std::get<1>(t5) << std::endl;
    std::cout << std::get<0>(t1) << std::endl;
    return 0;
}

// 获取tuple的值   std::get(std::tuple)
#include <iostream>
#include <tuple>

int main() {
    auto t1 = std::make_tuple(1, "PI", 3.14);
    std::cout << "(" << std::get<0>(t1) << "," << std::get<1>(t1)
              << "," << std::get<2>(t1) << ")"
              << "\n";
    return 0;
}

// std::tie()    // 解包函数
#include <iostream>
#include <tuple>

int main() {
    auto t1 = std::make_tuple(1, "PI", 3.14);
    auto t2 = std::make_tuple(2, "MAX", 999);
    std::cout << '(' << std::get<0>(t1) << ',' << std::get<1>(t1)
              << ',' << std::get<2>(t1) << ')' << '\n';
    int num = 0;
    std::string name;
    double value;
    std::tie(num, name, value) = t1;
    std::cout << '(' << num << ',' << name << ',' << value << ')' << '\n';
    num = 0, name = "", value = 0;
    std::tie(std::ignore, name, value) = t2;
    std::cout << '(' << num << ',' << name << ',' << value << ')' << '\n';
    return 0;
}

// 获取元素个数
#include <iostream>
#include <tuple>

int main() {
    auto t1 = std::make_tuple(2, "MAX", 999, 888, 65.6, "sun");
    std::cout << "The t1 has elements: " << std::tuple_size<decltype(t1)>::value << '\n';
    return 0;
}

// 获取元素类型
#include <iostream>
#include <tuple>

int main() {
    auto t1 = std::make_tuple(2, "MAX", 999.9);
    std::cout << "The t1 has elements: " << std::tuple_size<decltype(t1)>::value << '\n';
    std::tuple_element<0, decltype(t1)>::type type0;
    std::tuple_element<1, decltype(t1)>::type type1;
    std::tuple_element<2, decltype(t1)>::type type2;
    std::cout << type0 << std::endl;
    std::cout << type1 << std::endl;
    std::cout << type2 << std::endl;
    type0 = std::get<0>(t1);
    // type0 = 1.28;
    type1 = std::get<1>(t1);
    // type2 = std::get<2>(t1);
    type2 = 23.66666666;
    std::cout << "type0 : " << type0 << "\n";
    std::cout << "type1 : " << type1 << "\n";
    std::cout << "type2 : " << type2 << "\n";
    return 0;
}

// 使用 tuple 引用来改变tuple内元素的值
#include <iostream>
#include <tuple>
#include <functional>

int main() {
    auto t1 = std::make_tuple("test", 85.8, 185);
    std::string name;
    double weight;
    int height;
    auto t2 = std::make_tuple(std::ref(name), std::ref(weight), std::ref(height)) = t1;
    std::cout << "Before change : " << '\n';
    std::cout << '(' << name << ' ' << weight << ' ' << height << ')' << '\n';

    name = "spark", weight = 95.6, height = 188;
    std::cout << "After change : " << '\n';
    std::cout << '(' << std::get<0>(t2) << ' ' << std::get<1>(t2) << ' ' << std::get<2>(t2) << ')' << '\n';
    std::cout << '(' << std::get<0>(t1) << ' ' << std::get<1>(t1) << ' ' << std::get<2>(t1) << ')' << '\n';
    return 0;
}


#include <iostream>
#include <tuple>

int main() {
    auto temp = std::make_tuple(1760, 16, 1760, false, false);
    std::cout << std::get<0>(temp) << std::endl;
    std::cout << std::get<1>(temp) << std::endl;
    std::cout << std::get<2>(temp) << std::endl;
    std::cout << std::get<3>(temp) << std::endl;
    std::cout << std::get<4>(temp) << std::endl;
    return 0;
}
*/


/*
//   ====== 容器库 -- std::vector =======    //
//   https://blog.csdn.net/qq_44778120/article/details/122775457   //
// 成员函数
#include <iostream>
#include <vector>
using namespace std;
void assign_demo()
{
    vector<char> characters;
    vector<char> characters2;
    characters.assign(5, 'a');
    characters2.assign(characters.begin(), characters.begin() + 2);
    cout << "characters.begin()" << *characters.begin() << endl;
    for (char c : characters)
    {
        cout << c << '\n';
    }
    cout << "end of characters" << endl;
    for (char c : characters2)
    {
        cout << c << endl;
    }
    cout << "before initializer characters2" << endl;
    initializer_list<char> ilist = {'c', 'd', 'f'};
    characters.assign(ilist);
    for (char c : characters) {
        cout << c << '\n';
    }
}
int main(){
    assign_demo();
    return 0;
}

// 元素访问
#include <iostream>
#include <vector>
using namespace std;

void visit_demo() {
    vector<int> numbers{2, 4, 6, 8};
    cout << "Second element: " << numbers.at(1) << '\n';
    cout << "Second element: " << numbers[1] << '\n';
    cout << "First element: " << numbers.front() << endl;    // c.front() 等价于 *c.begin()
    cout << "First element: " << *numbers.data() << endl;
    cout << "Last element: " << numbers.back() << endl;

    numbers.at(1) = 3;
    numbers[2] = 5;
    numbers.front() = 0;
    *numbers.data() = 1;
    numbers.back() = 7;

    cout << "All numbers: ";
    for (auto i : numbers) {
        cout << ' ' << i;
    }
    cout << '\n';
}
int main() {
    visit_demo();
    return 0;
}

// 迭代器
#include <iostream>
using namespace std;
#include <vector>
#include <iterator>
#include <string>
void iterator_demo() {
    vector<int> ints{1, 2, 4, 8, 16};
    vector<string> fruits{"orange", "apple", "raspberry" };
    vector<char> empty;

    cout << "ints: ";
    for (vector<int>::const_iterator it = ints.cbegin(); it != ints.cend(); it++)
        cout << *it << " ";
    cout << endl;

    cout << "reverse of ints: ";
    for (auto it = ints.rbegin(); it != ints.rend(); ++it)
        cout << *it << " ";
    cout << endl;

    // 打印 vector fruits 中的首个水果，而不检查是否有一个
    cout << "First fruit: " << *fruits.begin() << "\n";

    if (empty.begin() == empty.end())
        cout << "vector 'empty' is indeed empty.\n";
}
int main() {
    iterator_demo();
    return 0;
}

// 容量
#include <iostream>
using namespace std;
#include <vector>
#include <iterator>
#include <string>
void capacity_demo() {
    cout << boolalpha;  // 启用流 str 中的 boolalpha 标志
    vector<int> numbers;
    cout << "Initially, numbers.empty(): " << numbers.empty() << '\n';
    numbers.push_back(42);
    cout << "After adding elements, numbers.empty(): " << numbers.empty() << '\n';

    vector<int> nums{1, 3, 5, 7};    //  size ：返回容纳的元素数
    cout << "nums contains " << nums.size() << " elements. \n";

    vector<char> s;     //  max_size :返回可容纳的最大元素数
    cout << "Maximum size of a 'vector' is " << s.max_size() << "\n";

    int sz = 100;

    cout << "not using reserve: \n";
    vector<int> v1;
    v1.push_back(10);
    v1.push_back(10);
    cout << sizeof(v1) << " bytes" << endl;
    cout << "Capacity is " << v1.capacity() << endl;    // capacity ：返回当前存储空间能够容纳的元素数

    std::cout << "using reserve: \n";
    v1.reserve(sz);                         // reserve ：预留存储空间
    cout << sizeof(v1) << " bytes" << endl;
    cout << "Capacity is " << v1.capacity() << endl;

    v1.shrink_to_fit();    // shrink_to_fit ：通过释放未使用的内存减少内存的使用
    cout << "Capacity after shrink_to_fit() is " << v1.capacity();
    cout << endl;
}
int main() {
    capacity_demo();
    return 0;
}
*/

// 修改器
#include <iostream>
using namespace std;
#include <vector>
#include <iterator>
#include <string>

template<typename T>
void print_vec(vector<T> vect)
{
    for (auto its = vect.cbegin(); its != vect.cend(); its++)
        cout << *its << " ";
    cout << endl;
}

void revise_demo() {
    vector<int> vec{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    print_vec(vec);
    vec.erase(vec.begin()); // 1
    print_vec(vec);
    vec.erase(vec.begin() + 2, vec.begin() + 5); // 2
    print_vec(vec);

    vec.pop_back();
    cout << "After using pop_back: ";
    print_vec(vec);

    vec.clear();
    cout << "After using clear:";
    print_vec(vec);

    vec = vector<int>(3, 100);
    auto it = vec.begin();
    it = vec.insert(it, 200); // 在it位置插入200， 返回begin()
    vec.insert(it, 2, 300);  // 在it位置插入两个300
    print_vec(vec);
    // "it" 不再合法，获取新值：
    it = vec.begin();

    vector<int> vec2(2, 400);
    vec.insert(it + 2, vec2.begin(), vec2.end()); // 在it+2的位置插入vec2
    print_vec(vec);

    int arr[] = {501, 502, 503};
    vec.insert(vec.begin(), arr, arr + 3);  // 在begin()处插入arr
    print_vec(vec);

    struct President
    {
        string name;
        string country;
        int year;

        President(string p_name, string p_country, int p_year)
            : name(move(p_name)), country(move(p_country)), year(p_year)
        {
            cout << "I am being constructed. \n"; // 构造函数
        }
        President(President&& other)
            : name(move(other.name)), country(move(other.country)), year(other.year)
        {
            cout << "I am being moved. \n"; // 拷贝构造函数
        }
        President &operator=(const President &other) = default;
    };
    vector<President> elections;
    cout << "emplace_back: \n";
    elections.emplace_back("Nelson Mandela", "South Africa", 1994); // 不会拷贝构造函数
    vector<President> reElections;
    cout << "\npush_back: \n";
    reElections.push_back(President("Franklin Delano Roosevelt", "the USA", 1936));  // 会调用拷贝构造函数

    vector<int> c = {1, 2, 3};
    vector<int> c2 = {4, 5, 6};
    cout << "\nThe vector holds: ";
    print_vec(c);
    c.resize(5);
    cout << "After resize up to 5: ";
    print_vec(c);
    c.resize(2);
    cout << "After resize down to 2: ";
    print_vec(c);

    c.swap(c2);
    cout << "After swap with c2: ";
    print_vec(c);
}

int main() {
    revise_demo();
    return 0;
}
