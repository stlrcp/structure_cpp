#include <iostream>

extern "C" {
    void dependent_function() {
        std::cout << "This function depends on libdependency.so" << std::endl;
    }
}
