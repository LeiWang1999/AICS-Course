//
// Created by root on 2020/12/21.
//

#include "binary_adder.h"
#include <climits>
void half_adder(bool &a, bool &b, bool &carry, bool &sum){
    sum = a ^ b;
    carry = a & b;
}

void full_adder(bool &carry_in, bool &a, bool &b, bool &carry, bool &sum){
    bool carry1,sum1,carry2;
    half_adder(carry_in,a,carry1,sum1);
    half_adder(sum1,b,carry2, sum);
    carry = carry1 | carry2;
}

std::string binary_adder(std::string a, std::string b, bool carry_in){
    std::string result;
    uint16_t a_len = a.length();
    uint16_t b_len = b.length();
    uint16_t max_len = 0;
    int16_t dif = a_len - b_len;
    if (dif>0){
        // inset 0 to b
        max_len = a_len;
        for (uint16_t i = 0; i< dif; i++){
            b.insert(0,"0");
        }
    }else{
        // inset 0 to a
        max_len = b_len;
        for (uint16_t i = 0; i< dif; i++){
            a.insert(0,"0");
        }
    }
    bool a_n[max_len];
    bool b_n[max_len];
    for (uint16_t i = 0; i< max_len; i++){
        a_n[i] = a[i]-'0';
        b_n[i] = b[i]-'0';
    }
    bool carry_out = 0, sum = 0;
    std::string insert_str = "";
    for (int16_t i = max_len - 1; i >=0 ; i--){
        full_adder(carry_in, a_n[i], b_n[i], carry_out, sum);
        carry_in = carry_out;
        insert_str = std::to_string(sum);
        result.insert(0, insert_str);
    }
    insert_str = std::to_string(carry_out);
    result.insert(0, insert_str);
    return result;
}