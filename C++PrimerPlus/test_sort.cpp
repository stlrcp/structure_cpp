/*
// 冒泡排序
#include <iostream>
using namespace std;
// 整数或浮点数皆可使用，若要使用类(class)或结构体(struct)时必须重载大于(>)运算符
template<typename T>
void bubble_sort(T arr[], int len) {
    int i, j;
    for (i = 0; i < len - 1; i++)
        for (j = 0; j < len - 1 - i; j++)
            if (arr[j] > arr[j+1])
                swap(arr[j], arr[j + 1]);
}
int main()
{
    int arr[] = {61, 17, 29, 22, 34, 60, 72, 21, 50, 1, 62};
    int len = (int)sizeof(arr) / sizeof(*arr);
    bubble_sort(arr, len);
    for (int i = 0; i < len; i++)
        cout << arr[i] << " ";
    cout << endl;
    float arrf[] = {17.5, 19.1, 0.6, 1.9, 10.5, 12.4, 3.8, 19.7, 1.5, 25.4, 28.6, 4.4, 23.8, 5.4};
    len = (float)sizeof(arrf) / sizeof(*arrf);
    bubble_sort(arrf, len);
    for (int i = 0; i < len; i++)
        cout << arrf[i] << ' ';
    cout << endl;
    return 0;
}


// 选择排序
#include <iostream>
#include <vector>
// 整数或浮点数皆可使用，若要使用类(class)时必须设定大于(>)的运算子功能
template<typename T>
void selection_sort(std::vector<T>&arr) {
    for (int i = 0; i < arr.size() - 1; i++) {
        int min = i;
        for (int j = i + 1; j < arr.size(); j++)
            if (arr[j] < arr[min])
                min = j;
        std::swap(arr[i], arr[min]);
    }
}
int main()
{
    std::vector<int> arr1{61, 17, 29, 22, 34, 60, 72, 21, 50, 1, 62};
    selection_sort(arr1);
    for (auto i : arr1) {
        std::cout << ' ' << i;
    }
    std::cout << '\n';
    std::vector<float> arr2{17.5, 19.1, 0.6, 1.9, 10.5, 12.4, 3.8, 19.7, 1.5, 25.4, 28.6, 4.4, 23.8, 5.4};
    selection_sort(arr2);
    for (auto i : arr2) {
        std::cout << ' ' << i;
    }
    std::cout << '\n';
    return 0;
}


// 插入排序
#include <iostream>
template<typename T>
void insertion_sort(T arr[], int len){
    for (int i = 1; i < len; i++){
        T key = arr[i];
        int j = i - 1;
        while ((j>=0) && (key<arr[j])){
                arr[j + 1] = arr[j];
                j--;
            }
        arr[j + 1] = key;
    }
}
int main()
{
    int arr1[] = {61, 17, 29, 22, 34, 60, 72, 21, 50, 1, 62};
    int len = (int)sizeof(arr1) / sizeof(*arr1);
    insertion_sort(arr1, len);
    for (int i = 0; i < len; i++)
        std::cout << arr1[i] << " ";
    std::cout << std::endl;
    float arr2[] = {17.5, 19.1, 0.6, 1.9, 10.5, 12.4, 3.8, 19.7, 1.5, 25.4, 28.6, 4.4, 23.8, 5.4};
    int len2 = (int)sizeof(arr2) / sizeof(*arr2);
    insertion_sort(arr2, len2);
    for (int i = 0; i < len; i++)
        std::cout << arr2[i] << " ";
    std::cout << std::endl;
    return 0;
}


// 希尔排序
#include <iostream>
template<typename T>
void shell_sort(T array[], int length)
{
    int h = 1;
    while (h < length / 3) {
        h = 3 * h + 1;
    }
    while (h >= 1) {
        for (int i = h; i < length; i++) {
            for (int j = i; j >= h && array[j] < array[j - h]; j -= h) {
                std::swap(array[j], array[j - h]);
            }
        }
        h = h / 3;
    }
}
int main()
{
    int arr1[] = {61, 17, 29, 22, 34, 60, 72, 21, 50, 1, 62};
    int len = (int)sizeof(arr1) / sizeof(*arr1);
    shell_sort(arr1, len);
    for (int i = 0; i < len; i++)
        std::cout << arr1[i] << " ";
    std::cout << std::endl;
    float arr2[] = {17.5, 19.1, 0.6, 1.9, 10.5, 12.4, 3.8, 19.7, 1.5, 25.4, 28.6, 4.4, 23.8, 5.4};
    int len2 = (int)sizeof(arr2) / sizeof(*arr2);
    shell_sort(arr2, len2);
    for (int i = 0; i < len2; i++)
        std::cout << arr2[i] << " ";
    std::cout << std::endl;
    return 0;
}


// 归并排序
// 1. 迭代版
#include <iostream>
template<typename T>  // 整数或浮点数皆可使用，若要使用物件（class）时必须设定"小于"(<)的运算子功能
void merge_sort(T arr[], int len) {
    T *a = arr;
    T *b = new T[len];
    for (int seg = 1; seg < len; seg+=seg) {
        for (int start = 0; start < len; start += seg + seg) {
            int low = start, mid = std::min(start + seg, len), high = std::min(start + seg + seg, len);
            int k = low;
            int start1 = low, end1 = mid;
            int start2 = mid, end2 = high;
            while (start1 < end1 && start2 <end2)
                b[k++] = a[start1] < a[start2] ? a[start1++] : a[start2++];
            while (start1 < end1)
                b[k++] = a[start1++];
            while (start2 < end2)
                b[k++] = a[start2++];
        }
        T *temp = a;
        a = b;
        b = temp;
    }
    if (a != arr) {
        for (int i = 0; i < len; i++)
            b[i] = a[i];
        b = a;
    }
    delete[] b;
}
int main()
{
    int arr1[] = {61, 17, 29, 22, 34, 60, 72, 21, 50, 1, 62};
    int len1 = (int)sizeof(arr1) / sizeof(*arr1);
    merge_sort(arr1, len1);
    for (int i = 0; i < len1; i++)
        std::cout << arr1[i] << " ";
    std::cout << std::endl;
    float arr2[] = {17.5, 19.1, 0.6, 1.9, 10.5, 12.4, 3.8, 19.7, 1.5, 25.4, 28.6, 4.4, 23.8, 5.4};
    int len2 = (int)sizeof(arr2) / sizeof(*arr2);
    merge_sort(arr2, len2);
    for (int i = 0; i < len2; i++)
        std::cout << arr2[i] << " ";
    std::cout << std::endl;
    return 0;
}


// 归并排序
// 递归版
#include <iostream>
#include <vector>
#include <limits>
template<typename T>
void Merge(std::vector<T> &Array, int front, int mid, int end) {
    // preconditions:
    // Array[front...mid] is sorted
    // Array[mid+1 ... end] is sorted
    // Copy Array[front ... mid] to LeftsubArray
    // Copy Array[mid+1 ... end] to RightSubArray
    std::vector<T> LeftSubArray(Array.begin() + front, Array.begin() + mid + 1);
    std::vector<T> RightSubArray(Array.begin() + mid + 1, Array.begin() + end + 1);
    int idxLeft = 0, idxRight = 0;
    LeftSubArray.insert(LeftSubArray.end(), std::numeric_limits<T>::max());
    RightSubArray.insert(RightSubArray.end(), std::numeric_limits<T>::max());
    // Pick min of LeftSubArray[idxLeft] and RightSubArray[idxRight], and put into Array[i]
    for (int i = front; i <= end; i++){
        if (LeftSubArray[idxLeft] < RightSubArray[idxRight]) {
            Array[i] = LeftSubArray[idxLeft];
            idxLeft++;
        }else{
            Array[i] = RightSubArray[idxRight];
            idxRight++;
        }
    }
}
template<typename T>
void MergeSort(std::vector<T> &Array, int front, int end) {
    if (front >= end)
        return;
    int mid = (front + end) / 2;
    MergeSort(Array, front, mid);
    MergeSort(Array, mid + 1, end);
    Merge(Array, front, mid, end);
}
int main()
{
    std::vector<int> arr1 = {61, 17, 29, 22, 34, 60, 72, 21, 50, 1, 62};
    int front1 = 0;
    int end1 = arr1.size();
    MergeSort(arr1, front1, end1);
    for (auto i : arr1)
        std::cout << i << " ";
    std::cout << std::endl;
    std::vector<float> arr2 = {17.5, 19.1, 0.6, 1.9, 10.5, 12.4, 3.8, 19.7, 1.5, 25.4, 28.6, 4.4, 23.8, 5.4};
    int front2 = 0;
    int end2 = arr2.size();
    // MergeSort(arr2, front2, end2);    // 有报错，内存问题？，暂未解决
    for (auto i : arr2)
        std::cout << i << " ";
    std::cout << std::endl;
    return 0;
}
*/
