//
// Created by root on 2020/12/21.
//

#include "binary_adder.h"

void half_adder(bool &a, bool &b, bool &carry, bool &sum){
    sum = a ^ b;
    carry = a & b;
}

void full_adder(bool &carry_in, bool &a, bool &b, bool &carry, bool &sum){
    bool carry1,sum1,carry2;
    half_adder(a,b,carry1,sum1);
    half_adder(sum1,b,carry2, sum);
    carry = carry1 | carry2;
}

void binary_adder(std::string a, std::string b, std::string & result, bool carry = 0){
    
}
