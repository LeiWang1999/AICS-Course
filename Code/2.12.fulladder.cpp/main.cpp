#include <iostream>
#include <string>
#include "data/binary_adder.h"

using namespace std;
int main(){
    string a = "1001";
    string b = "1";
    bool carry = 0;
    string result = "0";
    result = binary_adder(a , b, carry);
    cout << result << endl;
    return 0;
}