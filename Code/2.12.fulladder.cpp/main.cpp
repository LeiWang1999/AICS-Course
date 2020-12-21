#include <iostream>
#include "data/binary_adder.h"

using namespace std;
int main(){
    string a = "1001";
    string b = "0001";
    bool carry = 0;
    string result = "0";
//    half_adder(a, b, carry, sum);
//    full_adder(a,b,c,carry, sum);
    binary_adder("1001", "0001", carry=carry, result);
    std::cout << carry << sum << std::endl;
    return 0;
}