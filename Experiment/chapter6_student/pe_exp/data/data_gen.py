#!/usr/bin/env python
###################################################################
# File Name: data_gen.py
# Author: Li Zhen
# Email: lizhen2014@ict.ac.cn
# Created Time: 2020-02-13 09:24:58 CTS
# Description: 
###################################################################
from os import sys
import random as rd
import math as mh

def num2str(num, mode, width):
  str_ori = ""
  # bin
  if(mode == 0):
    str_ori = bin(num)[2:]
  else:
    str_ori = hex(num)[2:]

  str = ""
  if(len(str_ori) < width):
    str = (width - len(str_ori)) * "0" + str_ori
  else:
    str = str_ori

  return str

def verctor_gen(pfr, nfr, sfr, iter):
  iter0   = rd.randint(0,15)
  iter1   = rd.randint(0,15)
  
  line_iter = 32
  base      = 2**16
  width     = 4

  partsum = 0
  for i in range(iter):
    neu_str = ""
    syn_str = ""
  
    for k in range(line_iter):
      neu = rd.randint(0,2**16) % base
      syn = rd.randint(0,2**16) % base
      neu_str = neu_str + num2str(neu, 1, width) 
      syn_str = syn_str + num2str(syn, 1, width) 
  
      if(neu >= (base/2)):
        neu = neu - base
      if(syn >= (base/2)):
        syn = syn - base
      partsum += neu * syn      
    nfr.write(neu_str+"\n")
    sfr.write(syn_str+"\n")
  print("======================") 
  print(partsum) 
  print("======================") 
 
  # dump partial sum
  if(partsum < 0):
    partsum += 2**45
  
  pfr.write(num2str(partsum, 0, 45)+"\n")

#if __name__ == "__main_":
nfr = open("neuron", "w+")
sfr = open("weight", "w+")
pfr = open("result", "w+")

verctor_gen(pfr, nfr, sfr, 20)
verctor_gen(pfr, nfr, sfr, 30)
verctor_gen(pfr, nfr, sfr, 40)
verctor_gen(pfr, nfr, sfr, 50)

