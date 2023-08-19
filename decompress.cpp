#include<bits/stdc++.h>
using namespace std;
class Decompress{
    public:
    unordered_map<int, vector<int>> itemmap;
    vector<vector<int>> transactions;
    string fileLocation;
    string outputlocation;
    long long bufferlimit;
    long long buffersize = 0;
    queue<string> buffer;


    Decompress(string input,string output,int limit){
        fileLocation = input;
        outputlocation = output;
        bufferlimit = limit;
    }
    void write(){
        ofstream ofile(outputlocation,ios::app);
        if(!ofile.is_open())cerr<<"File is not opened"<<endl;
        int k = buffer.size();
        while(k--){
            string a = buffer.front();
            buffer.pop();
            ofile<<a<<endl;
        }
        ofile.close();
    }
    void buffer_insert(string& s){//Working fine
        if(buffersize+s.size()>bufferlimit){
            write();
            buffersize = 0;
        }
        buffer.push(s);
        buffersize+=s.size();
    }
    void Read(){
        ifstream transaction(fileLocation);
        if(!transaction.is_open())cerr<<"File is not opened"<<endl;
        if(!transaction.is_open())cerr<<"File is not opened second time."<<endl;
        bool ended = false;
        string s;
        while(getline(transaction,s)){
            if(!ended){
                vector<int> temp;
                stringstream obj(s);
                int var;
                while(obj>>var){
                    temp.push_back(var);
                    if(var==-1) ended = true;
                }
                if(!ended)transactions.push_back(temp);
            }
            else{
                vector<int> temp;
                stringstream obj(s);
                int var;
                obj>>var;
                int k = var;
                while(obj>>var)temp.push_back(var);
                itemmap[k] = temp;
            }
        }
        transaction.close();
    }
    void writetooutput(){
        int m = transactions.size();
        for(int i=0;i<m;i++){
            string temp;
            int n = transactions[i].size();
            for(int j=0;j<n;j++){
                if(transactions[i][j]>=0){
                    temp += to_string(transactions[i][j])+" ";
                    continue;
                }
                for(int k:itemmap[transactions[i][j]])temp += to_string(k)+" ";
            }
            buffer_insert(temp);
        }
        write();
    }
};
int main(string fileLocation,string outputlocation){
    Decompress decom(fileLocation,outputlocation,300000);
    decom.Read();
    decom.writetooutput();
}