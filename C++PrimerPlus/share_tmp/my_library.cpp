// my_library.cpp
#include <iostream>
#include "dependency.h"

extern "C" {  // 确保不进行 C++ 名称修饰
    // void dependent_function(){
    //     std::cout << "This function depends on libdependency.so" << std::endl;
    // }

    void my_library_function(){
        std::cout << "This is a function in my library" << std::endl;
        dependent_function();
    }
}
