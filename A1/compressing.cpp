#include<bits/stdc++.h>
using namespace std;
class FullNode{
    public:
    int val;
    int count = 0;
    int isLeaf = 0;
    FullNode* parent = NULL;
    bool exist = false;
    int key;
    map<int,FullNode*> childs;
    FullNode(int n=-1){
        val = n;
    }
    ~FullNode(){
        for(auto i:childs)delete i.second;
        childs.clear();
    }
};
class FullFPTree{
    public:
    FullNode* root;
    unordered_map<int,int> itemfreq;
    FullFPTree(){
        root = new FullNode();
    }
    ~FullFPTree(){
        delete root;
        itemfreq.clear();
    }
    void Insert(vector<int>& transaction){
        FullNode* curr = root;
        curr->count++;
        for(int i:transaction){
            if(!curr->childs.count(i))curr->childs[i] = new FullNode(i);
            curr->childs[i]->count++;
            curr->childs[i]->parent = curr;
            curr = curr->childs[i];
        }
        curr->isLeaf++;
    }
    void dfs(FullNode* i,vector<int>& temp,vector<vector<int>>& fina){
        temp.push_back(i->val);
        for(auto j:i->childs)dfs(j.second,temp,fina);
        if(!(int)i->childs.size())fina.push_back(temp);
        temp.pop_back();
    }
    void Print(){
        vector<int> temp;
        vector<vector<int>> fina;
        dfs(root,temp,fina);
        int n = fina.size();
        for(int i=0;i<n;i++){
            int m = fina[i].size();
            for(int j=0;j<m;j++)cout<<fina[i][j]<<" ";
            cout<<endl;
        }
    }
    void preload(string& fileLocation){
        ifstream transactions(fileLocation);
        if(!transactions.is_open())cerr<<"File is not opened"<<endl;
        string s;
        while(getline(transactions,s)){
            stringstream obj(s);
            int var;
            while(obj>>var)itemfreq[var]++;
        }
        transactions.close();
        ifstream transaction(fileLocation);
        if(!transaction.is_open())cerr<<"File is not opened"<<endl;
        if(!transaction.is_open())cerr<<"File is not opened second time."<<endl;
        while(getline(transaction,s)){
            vector<int> temp;
            stringstream obj(s);
            int var;
            while(obj>>var)temp.push_back(var);
            sort(temp.begin(),temp.end(),[this](int& a, int& b){return itemfreq[a]!=itemfreq[b]?itemfreq[a]>itemfreq[b]:a>b;});
            Insert(temp);
        }
        transaction.close();
    }
};
class Node{
    public:
    int val;
    int count = 0;
    Node* parent = NULL;
    Node* next = NULL;
    map<int,Node*> childs;
    Node(int n=-1){
        val = n;
    }
    ~Node(){
        for(auto i:childs)delete i.second;
        childs.clear();
    }
};
class FPTree{
    public:
    Node* root;
    unordered_map<int,int> itemfreq;
    map<pair<int,int>, Node*> headerFile;
    FPTree(){
        root = new Node();
    }
    ~FPTree(){
        delete root;
        itemfreq.clear();
        headerFile.clear();
    }
    void ConditionalInsert(vector<int>& transaction,int threshold,int c){
        Node* curr = root;
        curr->count+=c;
        for(int i:transaction){
            if(itemfreq[i]<threshold)continue;
            if(!curr->childs.count(i)){
                Node* newnode = new Node(i);
                curr->childs[i] = newnode;
                if(!headerFile.count(make_pair(itemfreq[i],i)))headerFile[make_pair(itemfreq[i],i)]=newnode;
                else{
                    Node* headercurr = headerFile[make_pair(itemfreq[i],i)];
                    while(headercurr->next)headercurr = headercurr->next;
                    headercurr->next = newnode;
                }
            }
            curr->childs[i]->count+=c;
            curr->childs[i]->parent = curr;
            curr = curr->childs[i];
        }
    }
    void dfs(Node* i,vector<int>& temp,vector<vector<int>>& fina){
        temp.push_back(i->val);
        for(auto j:i->childs)dfs(j.second,temp,fina);
        if(!(int)i->childs.size())fina.push_back(temp);
        temp.pop_back();
    }
    void Print(){
        vector<int> temp;
        vector<vector<int>> fina;
        dfs(root,temp,fina);
        int n = fina.size();
        for(int i=0;i<n;i++){
            int m = fina[i].size();
            for(int j=0;j<m;j++)cout<<fina[i][j]<<" ";
            cout<<endl;
        }
    }
    void ConditionalPreload(string& fileLocation,int threshold){
        ifstream transactions(fileLocation);
        if(!transactions.is_open())cerr<<"File is not opened"<<endl;
        string s;
        while(getline(transactions,s)){
            stringstream obj(s);
            int var;
            while(obj>>var){
                itemfreq[var]++;
            }
        }
        transactions.close();
        ifstream transaction(fileLocation);
        if(!transaction.is_open())cerr<<"File is not opened"<<endl;
        if(!transaction.is_open())cerr<<"File is not opened second time."<<endl;
        while(getline(transaction,s)){
            vector<int> temp;
            stringstream obj(s);
            int var;
            while(obj>>var)temp.push_back(var);
            sort(temp.begin(),temp.end(),[this](int& a, int& b){return itemfreq[a]!=itemfreq[b]?itemfreq[a]>itemfreq[b]:a>b;});
            ConditionalInsert(temp,threshold,1);
        }
        transaction.close();
    }
};
class CompressorDecompressor{
    public:
    FullFPTree* Tree;
    FPTree* ConTree;
    string fileLocation;
    vector<vector<int>> FrequentItemSet;
    unordered_map<int,int> InsertedKeys;
    string outputlocation;
    int presentkey = -2;
    queue<vector<int>> outputs;

    int Generatename(int i){
        if(InsertedKeys.count(i))return InsertedKeys[i];
        InsertedKeys[i] = presentkey;
        presentkey--;
        return InsertedKeys[i];
    }

    CompressorDecompressor(string input,string output){
        fileLocation = input;
        outputlocation = output;
        if (remove(output.c_str()) != 0);  
        Tree = new FullFPTree();
        ConTree = new FPTree();
    }
    void write(){
        ofstream ofile(outputlocation,ios::app);
        if(!ofile.is_open())cerr<<"File is not opened"<<endl;
        int k = outputs.size();
        while(k--){
            vector<int> a = outputs.front();
            outputs.pop();
            for(int j:a)ofile<<j<<" ";
            ofile<<endl;
        }
        ofile<<-1<<endl;
        for(auto i: InsertedKeys){
            ofile<<i.second<<" ";
            for(int j:FrequentItemSet[i.first])ofile<<j<<" ";
            ofile<<endl;
        }
        ofile.close();
    }
    FPTree* BuildConditionalTree( Node* end,int threshold){//checked working fine but getting some difference with rutvik values.
        FPTree* ConditionalTree = new FPTree();
        Node* endcurr = end;
        while(endcurr){
            Node* curr = endcurr;
            curr = curr->parent;
            while(curr->val!=-1){
                ConditionalTree->itemfreq[curr->val]+=endcurr->count;
                curr = curr->parent;
            }
            endcurr = endcurr->next;
        }
        endcurr = end;
        while(endcurr){
            Node* curr = endcurr;
            curr = curr->parent;
            vector<int> transaction;
            while(curr->val!=-1){
                transaction.push_back(curr->val);
                curr = curr->parent;
            }
            sort(transaction.begin(),transaction.end(),[&ConditionalTree](const int& a,const int& b){return ConditionalTree->itemfreq[a]!=ConditionalTree->itemfreq[b]? ConditionalTree->itemfreq[a]>ConditionalTree->itemfreq[b]:a>b;});
            ConditionalTree->ConditionalInsert(transaction,threshold,endcurr->count);
            endcurr = endcurr->next;
        }
        return ConditionalTree;
    }
    void Mine(FPTree*& contree,int threshold,vector<int>& prefix){//Checked Working Fine.
        auto it = contree->headerFile.begin();
        while(it!=contree->headerFile.end()){
            Node* curr = it->second;
            FPTree* ConditionalTree = BuildConditionalTree(curr,threshold);
            prefix.push_back(curr->val);
            Mine(ConditionalTree,threshold,prefix);
            delete ConditionalTree;
            if((int)prefix.size()>2)FrequentItemSet.push_back(prefix);
            prefix.pop_back();
            it++;
        }
    }
    void FrequentItemSetInsertdfs(FullNode* curr,int i,int j){
        if(curr->val!=-1){
            int n = FrequentItemSet[i].size();
            if(j>=n || Tree->itemfreq[FrequentItemSet[i][j]]>Tree->itemfreq[curr->val])return;
            if(FrequentItemSet[i][j]==curr->val)j++;
            if(j==n){
                if(!curr->exist || (curr->exist && n>(int)FrequentItemSet[curr->key].size())){
                    curr->exist = true;
                    curr->key = i;
                }
                return;
            }
        }
        for(auto nbr:curr->childs)FrequentItemSetInsertdfs(nbr.second,i,j);
    }
    void FrequentItemSetInsert(){
        int n = FrequentItemSet.size();
        FullNode* curr = Tree->root;
        for(int i=0;i<n;i++){
            sort(FrequentItemSet[i].begin(),FrequentItemSet[i].end(),[this](const int& a,const int& b){return Tree->itemfreq[a]!=Tree->itemfreq[b]? Tree->itemfreq[a]>Tree->itemfreq[b]:a>b;});
            FrequentItemSetInsertdfs(curr,i,0);
        }
    }
    vector<int> CompressedTransaction(vector<int>& transaction,vector<int> keyslist){
        vector<int> ans;
        sort(keyslist.begin(),keyslist.end(),[this](const int& a,const int& b){return (int)FrequentItemSet[a].size()!=(int)FrequentItemSet[b].size()? (int)FrequentItemSet[a].size()>(int)FrequentItemSet[b].size():a>b;});
        unordered_map<int,int> insertedelements;
        int n = keyslist.size();
        for(int i : keyslist){
            bool disjoint = true;
            for(int j:FrequentItemSet[i])if(insertedelements.count(j)){
                disjoint = false;
                break;
            }
            if(!disjoint)continue;
            for(int j:FrequentItemSet[i])insertedelements[j]++;
            int name = Generatename(i);
            ans.push_back(name);
        }
        for(auto i:transaction){
            if(insertedelements.count(i) && insertedelements[i]>0){
                insertedelements[i]--;
                continue;
            }
            ans.push_back(i);
        }
        return ans;
    }
    void transactiondfs(FullNode* curr,vector<int>& transaction,vector<int>& keyslist){
        if(curr->val!=-1)transaction.push_back(curr->val);
        if(curr->exist)keyslist.push_back(curr->key);
        if(curr->isLeaf){
            vector<int> s = CompressedTransaction(transaction,keyslist);
            int k = curr->isLeaf;
            while(k--)outputs.push(s);
        }
        for(auto nbr : curr->childs)transactiondfs(nbr.second,transaction,keyslist);
        if(curr->val!=-1)transaction.pop_back();
        if(curr->exist)keyslist.pop_back();
    }
    int generatethreshold(){
        vector<int> v;
        for(auto i:Tree->itemfreq){
            v.push_back(i.second);
        }
        sort(v.begin(),v.end(),greater<int>());
        int temp=v.size();
        temp = 0.2*temp;
        return v[temp];
    }
    void Compressor(int threshold){
        ConTree->ConditionalPreload(fileLocation,threshold);
        vector<int> prefix;
        Mine(ConTree,threshold,prefix);
        delete ConTree;
        FrequentItemSetInsert();
        vector<int> transaction;
        vector<int> keyslist;
        transactiondfs(Tree->root,transaction,keyslist);
        write();
    }
};

int main(int argc ,char* argv[]){
    string s=argv[1];
    string out =argv[2];
    CompressorDecompressor com(s,out);
    com.Tree->preload(s);
    cout<<com.generatethreshold()<<endl;
    com.Compressor(com.generatethreshold());
    cout<<com.FrequentItemSet.size()<<endl;
    return 0;
}
