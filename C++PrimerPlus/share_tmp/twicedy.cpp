#include "dependency.h"
#include "my_library.h"
#include <iostream>

extern "C" { 
    void twicedy_function(){
        std::cout << "This is the twicedy_function!!!!" << std::endl;
        dependent_function();
        my_library_function();
    }
}
