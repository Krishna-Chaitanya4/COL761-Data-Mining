#include <bits/stdc++.h>
using namespace std;

int preload(string fileLocation ){
        int n=0;
        ifstream transactions(fileLocation);
        if(!transactions.is_open())cerr<<"File is not opened"<<endl;
        string s;
        while(getline(transactions,s)){
            stringstream obj(s);
            int var;
            while(obj>>var){
                n++;
            }
        }
return n;

}
int main(){
int m=preload("/home/groot/Desktop/Coding/A1_datasets/D_small.dat");
int n=preload("/home/groot/Desktop/Coding/A1_datasets/output.dat");

cout<<(n*100/m)<<endl;

return 0;

}