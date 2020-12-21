//
// Created by root on 2020/12/21.
//

#ifndef INC_2_12_FULLADDER_CPP_BINARY_ADDER_H
#define INC_2_12_FULLADDER_CPP_BINARY_ADDER_H

#include <string>

void half_adder(bool &a, bool &b, bool &carry, bool &sum);
void full_adder(bool &carry_in, bool &a, bool &b, bool &carry, bool &sum);
void binary_adder(std::string a, std::string b, std::string & result, bool carry = 0);
#endif //INC_2_12_FULLADDER_CPP_BINARY_ADDER_H
