from scipy import sparse
import numpy as np
import pandas as pd
import csv 
import sys
import math
import operator

def classify_data(likeli,priori):
    with open('testing.csv') as f:
        reader = csv.reader(f)
        likeliArray = np.zeros(shape=(20,61188));
        for k in range(len(likeliArray)):
            for m in range(len(likeliArray[0])):
                likeliArray[k][m] = likeli[k][m] * priori[k];
        for row in reader:
            vals = np.zeros(20)
            word = map(float,row)
            firstElem = int(word.pop(0))
            vals = np.dot(likeliArray,word);
            # for k in range(int(19)):
#                 like = map(float,likeli[k,:])
#                 #vals[k] = sum([i+j+priori[k] for i, j in zip(like,word)])
#                 vals[k] = np.dot(like,word)*priori[k];
            index, value = max(enumerate(vals), key=operator.itemgetter(1))
            print (str(firstElem) + ', ' + str(index+1));

def calc_likeli(uniqWord, classArr, beta):
    likeliArray = np.zeros(shape=(20,61188));
    for i in range(len(classArr)):
        for j in range(len(classArr[0])):
            likeliArray[i][j] = (classArr[i][j] + (beta - 1))/(uniqWord[i] + (61188 * beta - 1));
    return likeliArray;

def calc_priori(data, occurArray):
    for i in range(len(occurArray)):
        occurArray[i] = occurArray[i]/12000.00;
    return occurArray;

def uniqueWords(newsGroups):
    uniqArr = np.zeros(20);
    for i in range(len(newsGroups)):
        for j in range(len(newsGroups[0])):
            if(newsGroups[i][j] > 0):
                uniqArr[i] += 1;
    return uniqArr;

if __name__== "__main__":
    clas = 0;
    occurArray = np.zeros(20);
    likeliArr = np.zeros(shape=(20,61188));
    classArray = np.zeros(shape=(20,61188));
    uniqArr = np.zeros(20);
    #data = pd.read_csv("training.csv");
    #data = np.array(list(csv.reader(open("training.csv", "rb"), delimiter=","))).astype("int");
    reader = csv.reader(open('training.csv', 'r'), delimiter=",")
    data = list(reader)
    print("Done reading file");
    for i in range (len(data)):
        clas = int(data[i][len(data[0])-1]);                
        occurArray[clas-1]+=1;
        for j in range (1,len(data[0])-2):
            if(data[i][j] != '0'): 
                classArray[clas-1][j] += 1;
    occurArray = calc_priori(data,occurArray);
    del(data);
    uniqArr = uniqueWords(classArray);
    likeliArr = calc_likeli(uniqArr,classArray,(1/61188));
    classify_data(likeliArr,occurArray);
